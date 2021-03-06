#from distutils.core import setup, Extension
from setuptools import setup

setup(name='PenguTrack',
      version='0.2',
      description='Extendable Toolbox for object tracking in image analysis',
      url='https://bitbucket.org/fabry_biophysics/pengu_track',
      author='Alexander Winterl',
      author_email='alexander.winterl@fau.de',
      packages = ['PenguTrack'],
      #py_modules=['PenguTrack','PenguTrack.Detectors','PenguTrack.Filters', 'PenguTrack.Models',
      #            'PenguTrack.Tools.Correction','PenguTrack.Tools.Eval_Detections','PenguTrack.Tools.Example','PenguTrack.Tools.PlotDetectionEvaluation',
      #            'PenguTrack.Tools.Rotate', 'PenguTrack.Tools.rotateMarker'],
      install_requires=['numpy','scipy','scikit-image','matplotlib','peewee','pandas']
      )
