import argparse
import atexit
from collections import deque
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
    seed=None,
    )

history = None
input_queue = deque()

# First parsing

# First we parse only the options that would load other configuration options,
# and update default parameters (e.g. --config).  Then we will re-parse all
# arguments like --model which can override the loaded arguments.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', '-c', default='~/.config/llm-cli.yaml')
parser.add_argument('--thread', '-t')
args, remaining = parser.parse_known_args()

# Config file if given
if args.config:
    args.config = os.path.expanduser(args.config)
    if os.path.exists(args.config):
        params.update(yaml.safe_load(open(args.config)))

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


# Second round parsing.  This re-parses --config and --thread from above but
# doesn't use them anymore.
parser = argparse.ArgumentParser(
    description="Defaults listed below include any settings in the "
                "config file or loaded history files."
    )
parser.add_argument('--list-models', '-l', action='store_true')
parser.add_argument('--thread', '-t', help="Read/save history&parameters from/to this file (yaml format)")
parser.add_argument('--system', '-s', default=params['system'],
                    help="The system prompt (default %(default)s)")
parser.add_argument('--model', '-m', default=params['model'],
                    help="Model name to select (default %(default)s)")
parser.add_argument('--temperature', '-T', type=float, default=params['temperature'],
                    help="Model temperature (default %(default)s)")
parser.add_argument('--max-tokens', default=params['max_tokens'], type=int,
                    help="Max input tokens to the model (save some for the output).  The config file name of this is max_tokens  (default %(default)s).")
parser.add_argument('--replay-history', action='store_true',
                    help='Replay each user prompt back through the model to update assistant prompts')
parser.add_argument('--no-interact', action='store_true',
                    help='Exit as soon as automatic input is done')
parser.add_argument('--seed', type=int, default=params['seed'],
                    help="Send this seed to the API (if unset, don't send anything")
parser.add_argument('--config', '-c', default='~/.local/llm.yaml',
                    help="Standard config options (default %(default)s)")
parser.add_argument('--verbose', '-v', action='store_true',
                    help="Be more verbose")
parser.add_argument('--search',
                    help="Search this and add to context window.  Argument can be @filename to load. (default is no search)")
parser.add_argument('--search-module', default='__main__.search_scicomp_docs',
                    help="Module to use for searching (default %(default)s)")
parser.add_argument('query', nargs='*',
                    help="Add these as queries.  Each query should be quoted and is sent in sequence.  ")
args = parser.parse_args()
# Replay all inputs of the history back to the model.
if args.replay_history:
    for message in history:
        if message['role'] == 'user':
            input_queue.append(message['content'])
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



def search_scicomp_docs(query, snipets=False, limit=5, **kwargs):
    """Example search function.

    Arguments: one, the query.  Can accept other argumest, and should
    accept arbitrary arguments via **kwargs in case other arguments are
    added later.

    Returns: iterator over dicts that should have at least these keys:
       ref: link or reference to the respective information
       text: Plain text or markdown information.
    """
    r = requests.get('https://scicomp-docs-search.k8s-test.cs.aalto.fi/', params={'q': query})
    for result in r.json():
        text = result['body'].strip()
        yield {'ref': result['path'], 'text': text}



def search_chroma(query, db=None, limit=5, **kwargs):
    """Search function for a Chroma database.

    This still isn't general and has some hard-codings.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,
    )
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db, embedding_function=embedding_function)
    docs = db.similarity_search(query, k=int(limit))
    #print(docs[0].page_content)
    #print(docs[0].metadata)
    for doc in docs:
        yield {'ref': doc.metadata.get('source', '-'), 'text': doc.page_content}



# Retrieval-augmented generation via --search and --search-module
if args.search:
    # Process the search module argument.  This is the function that lets us
    # retrieve data.
    # Format of search_mod:  module.function:arg1=value1,arg2=value2
    # See search_scicomp_docs above for an example of this function.
    search_mod = args.search_module
    # Get arguments if there are any
    if ':' in search_mod:
        search_mod, search_args = search_mod.split(':', 1)
        search_args = { arg.split('=',1)[0]: arg.split('=',1)[1] for arg in search_args.split(',') }
    else:
        search_args = { }
    # Split the module into the module and function name.  Import the function
    # as search_func.
    search_mod, search_func = search_mod.rsplit('.', 1)
    search_mod = __import__(search_mod, globals(), locals(), [search_func], 0)
    search_func = getattr(search_mod, search_func)

    query = args.search
    if query.startswith('@'):
        from langchain_community.document_loaders import UnstructuredFileLoader
        query = UnstructuredFileLoader(query[1:]).load()[0].page_content



    # Search and go through each result and add it to a results list.
    # Format it to include the reference and text body.
    results = [ ]
    results_chars = 0
    for result in search_func(query=query, **search_args):
        results.append(f"""From the reference {result['ref']} you have:\n{result['text'].strip()}""")
        results_chars += len(results[-1])
        # Limit the maximum number of searchable characters if specified.
        if 'max_chars' in search_args and results_chars > int(search_args['max_chars']):
            break
    results = "\n\n---\n\n".join(results)

    # Join everything together to the actual system prompt.  First, use the
    # main system prompt, then instructions to use the processed data, then the
    # results from the search query.
    params['system'] = f"""\
{params['system']}  You have the following information which you can use as part of your answers. Try to answer as correctly as possible.  If the information you need isn't in here, you should say that the search should be clarified rather than make things up.  A reference is given first, and then the information, you should include that reference when giving an answer.  The information follows:

---

{results}
"""



sess = requests.Session()
sess.auth=Auth()


# List models if given
if args.list_models:
    r = sess.get(urljoin(params['url'], 'models'))
    models = r.json()
    print("Models:")
    print(yaml.dump(models))
    exit(1)


# Manage command line history.  Copied from Python readline docs
histfile = os.path.join(os.path.expanduser("~"), ".llm_history")
try:
    readline.read_history_file(histfile)
    h_len = readline.get_current_history_length()
except FileNotFoundError:
    open(histfile, 'wb').close()
    h_len = 0
def _save_hist(prev_h_len, histfile):
    new_h_len = readline.get_current_history_length()
    readline.set_history_length(1000)
    readline.append_history_file(new_h_len - prev_h_len, histfile)
atexit.register(_save_hist, h_len, histfile)

# Main loop of running
if history is None:
    history = [ ]
print(f'System: {params["system"]} [{params["model"]}]')
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
    # Print history for user
    if data == r'\history':
        print(yaml.dump(history))
        continue
    # Force a save right now.
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
        print(r.status_code, r.reason)
        print(r.json())
        continue

    # Non-streamed responses
    if not params['stream']:
        rdata = r.json()
        #print(rdata)
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
