# llm.py

This serves as a command line shell to an OpenAI-like completion
service.  The purpose is to query the API from the command line.

Features include: saving per-thread history to topics (and re-loading
history from those states), history files are YAML so easy to edit and
re-test), use custom APIs (not just OpenAI), adjust all the main
parameters (model, temperature, system prompt), parameters per-thread
persisted to the thread files, global config file.


## Installation

`llm.py` is self-contained other than the fairly standard dependencies
`requests` and `pyyaml`.


## Usage

```
python llm.py [-t thread_history.yaml]
```

Options are loaded in this order, last one takes precedence.  Whatever
is loaded gets saved back to the history file:
* Build-in (see `params` dict)
* File specified with `--config`, `-c`
* Loaded from history with `-thread`, `-t`
* Specified on command line.

Options:

* `--thread`, `-t`: Save state to this YAML file and read from it.  It
  will also save some of the main parameters below.  This can be used
  to resume (and edit) chats and reply them.
* `--system`: The system prompt.
* `--temperature`, `--model`: as expected
* `--max-tokens`: Max model input tokens (save some for the output)
* `--config`: A config file with defaults.
* `--list-models`, `-l`: Don't do anything but list the models.

More options are explained by using `--help`.

The config file syntax is a YAML file containing a mapping from values
above to their values. (but `max_tokens` instead of `max-tokens`).


## Status

This was mainly made for one person's work and might not be that
useful to others (and partly made as an experiment for learning).
Still the threads part seems unique.  I'd welcome contributions or
feature requests.


## See also

* https://github.com/kardolus/chatgpt-cli - the first one I used
* https://github.com/simonw/llm - next thing I was recommended to
  use.  But didn't quite do what I want, at least I didn't see it when
  I looked.  **This might be what you are looking for instead.**
