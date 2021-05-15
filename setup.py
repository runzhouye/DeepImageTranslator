from cx_Freeze import setup, Executable

setup(name = "DIT" ,
      version = "1.0.0" ,
      description = "DESCRIPTION" ,
      executables = [Executable("DeepImageTranslator.py")]
)