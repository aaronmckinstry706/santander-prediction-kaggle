import pathlib
import pkgutil


def script_names():
    for module_info in pkgutil.walk_packages(path=[pathlib.Path(__file__).parent / 'scripts']):
        yield module_info.name


def test_script_names():
    scripts = [script for script in script_names()]
    assert 'sample_module_script' in scripts, 'module_script should be listed in the script names.'
    assert 'sample_package_script' in scripts, 'sample_package_script should be listed in the script names.'
    assert 'sample_package_script_main' not in scripts, ('sample_package_script_main is listed in script names, '
                                                         'but it should not be.')
