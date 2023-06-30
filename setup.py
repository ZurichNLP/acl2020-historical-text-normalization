from setuptools import setup

setup(name='contextual_transduction',
      version='0.1',
      description=('contextual transduction'),
      url='',
      author='Peter Makarov & Simon Clematide',
      author_email='makarov@cl.uzh.ch',
      license='MIT',
      packages=['contextualized_transduction'],
      install_requires=[
            "wheel==0.34.2",
            "dataclasses<=0.7",
            "editdistance==0.5.2",
            "numpy==1.16.1",
            "sacred==0.7.5",
            "pymongo==3.10.1",  # bug in sacred==0.7.5
            "scikit-learn==0.20.2",
            "scipy==1.2.0",
            "kenlm @ git+https://github.com/kpu/kenlm/#egg=kenlm",
            "nn_lm @ git+https://github.com/peter-makarov/nn_lms@master",
      ],
      package_data={
            "": ["data/med.txt", "data/train*.apra", "data/words.txt"],
      },
      zip_safe=True)
