from setuptools import setup

setup(name='smart-kube',
      package_dir={'': 'src'},
      version='0.0.1',
      install_requires=[
            'gym',
            'kubernetes',
            'pandas',
            'google.cloud.monitoring',
            'google.cloud.logging'
            ]
      )
