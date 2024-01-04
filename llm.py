import argparse
from collections import deque
import json
import os
from pathlib import Path
import readline
import sys
from urllib.parse import urljoin
import logging
import requests
import yaml

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# These are all the default parameters
params = dict(
    url='',
    api_key=os.environ.get('OPENAI_API_KEY', '321'),
    model='llama2-13b-chat',
    max_tokens=2048 * 3 // 4,
    system='You are a helpful assistant.',
    temperature=1,
    stream=True,
    seed=None,
    )

history = None
input_queue = deque()

# First parsing

# First we parse only the options that would load other configuration options,
# and update default parameters (e.g. --config).  Then we will re-parse all
# arguments like --model which can override the loaded arguments.

#TODO: Dont replace the default values with the config files -> 404 error
#TODO: Better error handeling
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', '-c', default='~/.local/llm.yaml')
parser.add_argument('--thread', '-t')
args, _ = parser.parse_known_args()

# Config file if given
if args.config:
    logging.info(f"Reading the config file from {args.config}")
    args.config = os.path.expanduser(args.config)
    if os.path.exists(args.config):
        params.update(yaml.safe_load(open(args.config)))
    else:
        logging.warning(f"No such file or directory!")

# Load saved thread if given
# thread_history is original file for updating (to
# not change these parameters if they aren't specified again)
thread_history = { }
if args.thread:
    args.thread = os.path.expanduser(args.thread)
    if os.path.exists(args.thread):
        thread_history = yaml.safe_load(open(args.thread))
        history = thread_history['history']
        keys = ['system', 'model', 'temperature', 'max_tokens', 'seed',
                'url', 'api_key', 'stream']
        for key in keys:
            if key in thread_history:
                params[key] = thread_history[key]

#TODO: Better logging and exception handling.
#TODO: Better config reading and args handeling
# Second round parsing.  This re-parses --config and --thread from above but
# doesn't use them anymore.
parser = argparse.ArgumentParser()
parser.add_argument('--list-models', '-l', action='store_true')
parser.add_argument('--thread', '-t', help="Read/save history&parameters from/to this file (yaml format)")
parser.add_argument('--system', '-s', default=params['system'],
                    help="The system prompt")
parser.add_argument('--model', '-m', default=params['model'])
parser.add_argument('--temperature', '-T', type=float, default=params['temperature'],
                    help="Model temperature")
parser.add_argument('--max-tokens', default=params['max_tokens'], type=int,
                    help="Max input tokens to the model (save some for the output).  The config file name of this is max_tokens.")
parser.add_argument('--replay-history', action='store_true',
                    help='Replay each user prompt back through the model to update assistant prompts')
parser.add_argument('--no-interact', action='store_true',
                    help='Exit as soon as automatic input is done')
parser.add_argument('--seed', type=int, default=params['seed'],
                    help="Send this seed to the API (if unset, don't send anything")
parser.add_argument('--config', '-c', default='~/.local/llm.yaml',
                    help="Standard config options")
parser.add_argument('--verbose', '-v', action='store_true',
                    help="Be more verbose")
parser.add_argument('query', nargs='*',
                    help="Add these as queries.  Each query should be quoted and is sent in sequence.  ")
args = parser.parse_args()
# Replay all inputs of the history back to the model.
if args.replay_history:
    logging.info(f"Replicating the history prompts.")
    for message in history:
        if message['role'] == 'user':
            input_queue.append(message['content'])
    args.no_interact = True
    history = None
if args.query:
    for query in args.query:
        input_queue.append(query)
    args.no_interact = True
params.update(vars(args))


# Utility functions
class Auth(requests.auth.AuthBase):
    """Requests authentication wrapper"""
    def __call__(self, r):
        r.headers['Authorization'] = f'Bearer {params["api_key"]}'
        return r

def Message(role, content):
    assert role in {'system', 'user', 'assistant'}
    return {'role': role, 'content': content}

#TODO: Better token counter
def count_tokens(text):
    """Number of tokens in some text"""
    return len(text) // 5

def limit_tokens(max_, messages):
    """History list -> limited history list with max_tokens.

    - Doesn't yet accurately count tokens.
    - Assumes that the first message is the system role.
    """
    system = messages[0]
    new = [ ]
    count = count_tokens(system['content'])
    for msg in reversed(messages[1:]):
        count += count_tokens(msg['content'])
        if count > max_:
            break
        new.append(msg)
    return [system] + list(reversed(new))

def save(fname):
    """Save history+parameters to fname
    """
    thread_history.update(dict(
        system=params['system'],
        model=params['model'],
        temperature=params['temperature'],
        max_tokens=params['max_tokens'],
        seed=params['seed'],
        history=history,
        ))
    open(fname+'.new', 'w').write(yaml.dump(thread_history))
    os.rename(fname+'.new', args.thread)



sess = requests.Session()
sess.auth=Auth()


# List models if given
if args.list_models:
    r = sess.get(urljoin(params['url'], 'models'))
    models = r.json()
    print("Models:")
    print(yaml.dump(models))
    exit(1)

def print_help():
    print(f"Type '\exit' or '\quit' to exit")
    print(f"Type '\save' to save the current history.")
    print(f"Type '\history' to show the current history.")
    print(f"Type '\help' or '\menu' to print this menu again.")

#TODO: How to exit?
# Main loop of running
if history is None:
    history = [ ]
print(f'System: {params["system"]} [{params["model"]}]')
print_help()
while True:
    print()
    if  input_queue:
        data = input_queue.popleft()
        print(f'>> {data}')
    elif args.no_interact:
        sys.exit()
    else:
        try:
            data = input('> ')
            if args.verbose:
                print(repr(data))
        except EOFError:
            break
    # No input, do nothing
    if not data.strip():
        continue

    if data.strip() in (r'\help', r'\menu'):
        print_help()
        continue

    if data.strip() in (r'\exit', r'\quit'):
        logging.info(f"Ending the chat interface with user command.")
        break

    # Print history for user
    if data == r'\history':
        print(yaml.dump(history))
        continue

    # Force a save right now.
    # TODO: Better thread handeling + add default path
    if data.startswith(r'\save'):
        args.thread = data.split(None, 1)[1].strip()
        save(args.thread)
        print(f'Saving history (now and future) to {args.thread}')
        continue

    history.append(Message('user', data))

    # Construct our API query.
    msg = {
        'model': params['model'],
        'temperature': params['temperature'],
        'stream': params['stream'],
        'messages': limit_tokens(params['max_tokens'], [
            {'role': 'system',
             'content': params['system']
            },
            ] + history),
        **({'seed': params['seed']} if params['seed'] else {})
        }
    if args.verbose:
        print(msg)

    # Post it and basic check.
    r = sess.post(urljoin(params['url'], 'chat/completions'), json=msg, stream=True)
    if r.status_code != 200:
        logging.warning(f"Connection failed wiht {r.status_code}. {r.reason}. {r.json}.")
        continue

    # Non-streamed responses
    if not params['stream']:
        rdata = r.json()

        rchoice = rdata['choices'][0]
        print(f"[{r.status_code}: {rdata['usage']['prompt_tokens']}→ {rdata['usage']['completion_tokens']}→ {rchoice['finish_reason']}]")
        message = rchoice['message']['content']
        print(message)
        print()

    # Streaming responses
    else:
        message = [ ]
        print(f"[{r.status_code}:]")
        finish_reason = 'no-finish-reason'
        for line in r.iter_content(chunk_size=None):
            for line in line.split(b'\r\n'):
                if not line: continue
                handled = False
                #print(line)
                # Handle metadata
                if line.endswith(b'[DONE]'):
                    break
                if line.startswith(b': ping -'):
                    continue
                try:
                    data = json.loads(line.split(b':', 1)[1])
                except:
                    print(line)
                    raise
                #print(data['choices'][0])
                if data['choices'][0]['delta'] == {'role': 'assistant'}:
                    continue
                if data['choices'][0]['finish_reason']:
                    finish_reason = data['choices'][0]['finish_reason']
                    handled = True
                    continue
                if 'content' not in data['choices'][0]['delta'] and not handled:
                    print(line)
                    continue
                delta = data['choices'][0]['delta']['content']
                print(delta, end='', flush=True)
                message.append(delta)

        print(f"[{finish_reason}]")
        print()
        message = ''.join(message)

    history.append(Message('assistant', message))

    # Save the conversation+parameters if we were given a thread file.
    if args.thread:
        save(args.thread)
