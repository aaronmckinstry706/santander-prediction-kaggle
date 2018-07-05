# script-project-template

A general project structure for working with several different (but related) Python scripts. 

## Work Flow

### Creating a script.

Create a script (either a module, `something_file.py`, or a package, `something_directory`) under the package 
`scripts`. The script must expose two functions, described below. To run the script, execute this from the top-level 
project directory:

`PYTHONPATH=. python run.py <script_name> [options...]`.

For example, given our module/package names in the previous paragraph, we would replace `<script_name>` with either 
`something_file` or `something_directory`. 

It'd probably be good to use a shell script to simplify the command to `run <script_name> [options...]`, but that's 
up to you. 

### Testing scripts.

The project is designed to run with `pytest`, with standard unit test discovery within files, and with the
`pytest.ini` file set for `pytest` to check every `.py` file in the project for unit tests. To run all tests, 
navigate to the top-level project directory and execute the `pytest` command. 

## Project Structure

Each test is either a module or a package directly under the `scripts` package. Each script exposes two methods:

* `add_args(arg_parser: argparse.ArgumentParser)`, and
* `run(clargs: argparse.Namespace)`. 

The method `add_args` adds any command line arguments for your script to the argument parser (do not parse the args, 
nor return a value). The method `run(...)` is the entry point to your script when it is called. 

## Additional Notes

When using files in unit tests, try to avoid using explicit directories. Use `get_data(..)` instead, and put your file 
resources under their own package within your script. 

If you want to (custom) type-check your CLI parameters, define a custom function which does the type-conversion as you 
want and throws an `argparse.ArgumentTypeError` whenever the value is technically the correct type, but not within 
the subset of values that you allow. 

## Git Usage

This project was intended to track experiments as scripts that could potentially be deliverables. As such, I have a 
specific use case in `git` that I would like to note. Feel free to ignore this if this isn't your use-case. 

When writing experiments in scripts (testing an optimization idea, or mathematical technique, etc.), I have a 
tendency to produce a cascade of related experiments, each based off the last (as, naturally, one idea leads to 
another, which leads to another, and so on); as a result, each corresponding script is based off the previous one. In
 `git` history, I want to track this flow. For example...

> I create an experiment script, run, and finalize the experiment. Now I have a new and related idea. Ah! I could
 use the old script and modify it! Hmmm, but then I would have to keep track of which commit had the old experiment 
 vs the new experiment. This is too much work for several experiments, and I want each file/directory to correspond to
  a single experiment. How do I do this?

It would be perfect if there was a `git cp` command--but, alas, this is 
not the case. The only way that I have found to do this is via the following series of commands, where `<exp1>` is 
the old experiment script and `<exp2>` is the new (similar, but soon-to-be-different) experiment script:

1. `git branch <exp2>`
2. `git mv <exp1> <exp2>` (still on non-`exp2` branch)
3. `git merge <exp2>`
4. In merge conflict, make sure to keep old `<exp1>` file, and commit. 

Ta-da! Now both the new and the old files have linked histories, which is what I wanted!

