from cx_Freeze import setup, Executable

setup(name = "hand writing generator" ,
      version = "0.1.0" ,
      description = "generates hand written svgs" ,
      executables = [Executable("GUI.py", base = "Win32GUI")]
)