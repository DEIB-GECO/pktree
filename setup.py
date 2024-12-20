from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize



# setup(
#     name = "pktree",
#     ext_modules = cythonize("hello.pyx")
# )

extensions = [Extension(name="pktree.hello", sources=["pktree/hello.pyx"])]

setup(
    name="pktree",
    version="0.1.0",
    #include_package_data=True,
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=["cython"]
)