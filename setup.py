from distutils.core import setup, Extension

setup(name='PenguTrack',
      version='0.2',
      py_modules=['PenguTrack','PenguTrack.Detectors','PenguTrack.Filters', 'PenguTrack.Models',
                  'Tools.Correction','Tools.Eval_Detections','Tools.Example','Tools.PlotDetectionEvaluation',
                  'Tools.Rotate', 'Tools.rotateMarker'],
      requires=['numpy','clickpoints','scipy','skimage','matplotlib','peewee','pandas'])