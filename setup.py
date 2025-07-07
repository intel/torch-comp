from setuptools import setup, find_packages

PACKAGE_NAME = "torch_comp"
setup(
    name=PACKAGE_NAME,
    version="0.1",
    description="compatible mode plugin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["torch", "ruamel.yaml", "pytest"],
    package_data={"torch_comp": ["yaml/*.yaml"]},
    include_package_data=True,
)
