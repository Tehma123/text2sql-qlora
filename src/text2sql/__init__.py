import os

# Must run before huggingface_hub is imported anywhere -- it caches this as
# a module-level constant (`huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER`,
# read once via os.environ.get() at that module's own import time), so
# setting it later (e.g. in text2sql.models.load_model, which runs after
# `import unsloth` has already pulled in transformers/huggingface_hub) has
# no effect. Every entry point imports some text2sql submodule first, which
# always executes this __init__.py before descending into it, so this is
# the one place early enough to guarantee it lands in time.
#
# Why disable it at all: Unsloth's default accelerated downloader
# (hf_transfer) hit a deterministic Windows bug in this environment -- it
# fails to clean up its own corrupted partial-download blob on retry
# ("Error while removing corrupted file: The process cannot access the file
# because it is being used by another process (os error 32)"), which fails
# every subsequent load of a not-yet-cached model, not just a one-off flake.
# setdefault() so an explicit override (e.g. on a machine where this doesn't
# reproduce) still wins.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
