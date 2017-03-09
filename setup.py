from distutils.core import setup, Extension

setup(name='PenguTrack',
      version='0.2',
      description='Extendable Toolbox for object tracking in image analysis',
      url='https://bitbucket.org/fabry_biophysics/pengu_track',
      author='Alexander Winterl',
      author_email='alexander.winterl@fau.de',
      py_modules=['PenguTrack','PenguTrack.Detectors','PenguTrack.Filters', 'PenguTrack.Models',
                  'PenguTrack.Tools.Correction','PenguTrack.Tools.Eval_Detections','PenguTrack.Tools.Example','PenguTrack.Tools.PlotDetectionEvaluation',
                  'PenguTrack.Tools.Rotate', 'PenguTrack.Tools.rotateMarker'],
      requires=['numpy','clickpoints','scipy','scikit-image','matplotlib','peewee','pandas'])
