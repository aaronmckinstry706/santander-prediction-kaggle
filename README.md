# [Santander Value Prediction Competition on Kaggle](https://www.kaggle.com/c/santander-value-prediction-challenge#description)

This repository contains all of my scripts which I use to explore the dataset, make predictions, and do anything else related to the project. 


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

The `master` branch will be for complete, deliverable scripts only. Any development work is being done in a Jupyter Notebook, and being committed to the `scratch` branch in the meantime so that we can save all of our precious work without fear of losing anything. Then, when we're ready to create a script, we move the script from the Jupyter Notebook and format it as a proper script in this project. 

When refactoring some Notebook code for use as a reproducible script, we first create and checkout a branch called `dev-<script-name>`. If you can't think of a script name immediately, just make it `dev` and then rename the branch later. Once the script is finished, merge it into `master`. 

Under *no* circumstances should you merge `scratch` into anywhere else. Other branches can be merged into `scratch`, but never the opposite. 

