from setuptools import setup

setup(
   name='stog',
   version='0.1',
   description='A semantics parsing framework',
   packages=['stog'],  #same as name
   install_requires=['overrides', 'torch==0.4.1', 'h5py', 'boto3', 'tensorboardX==1.2','spacy', 'ftfy', 'nltk', 'conllu']
)

# Write logs for training visualisation with the Tensorboard application
# Install the Tensorboard application separately (part of tensorflow) to view them.
