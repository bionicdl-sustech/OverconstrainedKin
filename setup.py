from setuptools import setup

setup(
   name='ov_kin',
   version='1.0',
   description='Package for overconstrained robot kinematics',
   author='Haoran Sun',
   author_email='hrsun@cs.hku.hk; 12150050@mail.sustech.edu.cn',
   packages=['ov_kin'],  #same as name
   install_requires=['numpy==1.26.0', 'modern-robotics', 'roboticstoolbox-python'], #external packages as dependencies
)