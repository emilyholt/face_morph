from distutils.core import setup

setup(
	name='Morphing',
	version='0.1dev',
	packages=['morphing',],
	install_requires=['numpy', 'dlib', 'scikit-image', 'opencv-python']
)