from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy



# setup(
#     name = "pktree",
#     ext_modules = cythonize("hello.pyx")
# )

## extensions = [Extension(name="pktree.hello", sources=["pktree/hello.pyx"])]
## 
## setup(
##     name="pktree",
##     version="0.1.0",
##     #include_package_data=True,
##     packages=find_packages(),
##     ext_modules=cythonize(extensions),
##     setup_requires=["cython"],
##     install_requires=["cython"],
## )

#ext-modules = [
#    {name = "pktree.hello", sources = ["pktree/hello.pyx"]},
#    {name = "pktree.tree._criterion", sources = ["pktree/tree/_criterion.pyx"], include-dirs=["numpy.get_include()"]},
#    {name = "pktree.tree._partitioner", sources = ["pktree/tree/_partitioner.pyx"]},
#    {name = "pktree.tree._splitter", sources = ["pktree/tree/_splitter.pyx"]},
#    {name = "pktree.tree._tree", sources = ["pktree/tree/_tree.pyx"]},
#   {name = "pktree.tree._utils", sources = ["pktree/tree/_utils.pyx"]},
#]

setup(
    #packages=find_packages(include=["pktree", "pktree.tree"]),
    packages=['pktree', 'pktree.tree'],
    package_data={'': ['*.pxd', '*.pyx']},
    ext_modules=cythonize(
        [Extension(
            "pktree.hello",
            sources=["pktree/hello.pyx"]
        ),
        Extension(
            "pktree.tree._criterion",
            sources=["pktree/tree/_criterion.pyx"],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            "pktree.tree._partitioner",
            sources=["pktree/tree/_partitioner.pyx"],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            "pktree.tree._splitter",
            sources=["pktree/tree/_splitter.pyx"],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            "pktree.tree._tree",
            sources=["pktree/tree/_tree.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++"
        ),
        Extension(
            "pktree.tree._utils",
            sources=["pktree/tree/_utils.pyx"],
            include_dirs=[numpy.get_include()]
        )]
    )
)