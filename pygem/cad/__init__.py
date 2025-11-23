"""PyGeM CAD init."""

try:
    pass
except ModuleNotFoundError as e:
    print("\nOCC not found, but required to deal with CAD files")
    print("Install it using:")
    print("\tconda install -c conda-forge pythonocc-core=7.4.0")
    print("or visit https://github.com/tpaviot/pythonocc-core for more info\n")
    raise e
