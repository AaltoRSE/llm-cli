import argparse
import json
import os
from pathlib import Path
import readline
import sys
from urllib.parse import urljoin

import requests
import yaml

# These are all the default parameters
params = dict(
    url='',
    api_key=os.environ.get('OPENAI_API_KEY', '321'),
    model='llama2-13b-chat',
    max_tokens=2048 * 3 // 4,
    system='You are a helpful assistant.',
    temperature=1,
    stream=True,
    )

history = None


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', '-c', default='~/.local/llm.yaml')
parser.add_argument('--thread', '-t')
args, remaining = parser.parse_known_args()
if args.config:
    params.update(yaml.safe_load(open(args.config)))
# Load saved thread if given
thread_history = { }
if args.thread and os.path.exists(args.thread):
    thread_history = yaml.safe_load(open(args.thread))
    history = thread_history['history']
    keys = ['system', 'model', 'temperature', 'max_tokens',
            'url', 'api_key', 'stream']
    for key in keys:
        if key in thread_history:
            params[key] = thread_history[key]
    #if not args.temperature and 'temperature' in data:
    #    params['temperature'] =  data['temperature']
    #if not args.system and 'system' in data:
    #    params.system = data['system']


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
parser.add_argument('--config', '-c', default='~/.local/llm.yaml',
                    help="Standard config options")
parser.add_argument('query', nargs='?')
args = parser.parse_args()
#if args.config:
#    params.update(yaml.safe_load(open(args.config)))
params.update(vars(args))


class Auth(requests.auth.AuthBase):
    def __call__(self, r):
        r.headers['Authorization'] = f'Bearer {params["api_key"]}'
        return r

def Message(role, content):
    return {'role': role, 'content': content}

def count_tokens(text):
    return len(text) // 5

def limit_tokens(max_, messages):
    system = messages[0]
    new = [ ]
    count = count_tokens(system['content'])
    for msg in reversed(messages[1:]):
        count += count_tokens(msg['content'])
        if count > max_:
            break
        new.append(msg)
    return [system] + list(reversed(new))


sess = requests.Session()
sess.auth=Auth()


# List models if given
if args.list_models:
    r = sess.get(urljoin(params['url'], 'models'))
    models = r.json()
    print("Models:")
    print(yaml.dump(models))
    exit(1)


# Main loop of running
if history is None:
    history = [ ]
print(f'System: {params["system"]} [{params["model"]}]')
while True:
    print()
    if args.query:
        data = args.query
    else:
        try:
            data = input('> ')
        except EOFError:
            break
    if data == r'\history':
        print(yaml.dump(history))
        continue

    history.append(Message('user', data))

    msg = {
        'model': params['model'],
        'temperature': params['temperature'],
        'stream': params['stream'],
        'messages': limit_tokens(params['max_tokens'], [
            {'role': 'system',
             'content': params['system']
            },
            ] + history)
        }

    r = sess.post(urljoin(params['url'], 'chat/completions'), json=msg, stream=True)
    if r.status_code != 200:
        print(r.status_code, r.reason)
        print(r.json())
        continue

    if not params['stream']:
        rdata = r.json()
        #print(rdata)
        rchoice = rdata['choices'][0]
        print(f"[{r.status_code}:  {rdata['usage']['prompt_tokens']}→ {rdata['usage']['completion_tokens']}→ {rchoice['finish_reason']}]")
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
    if args.thread:
        thread_history.update(dict(
            system=params['system'],
            model=params['model'],
            temperature=params['temperature'],
            max_tokens=params['max_tokens'],
            history=history,
            ))
        open(args.thread+'.new', 'w').write(yaml.dump(thread_history))
        os.rename(args.thread+'.new', args.thread)

    if args.query:
        break
