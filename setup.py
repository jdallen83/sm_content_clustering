from setuptools import setup


setup(
    name='sm_content_clustering',
    version='0.1',
    description='A Python module for clustering creators of social media content into networks',
    url='http://github.com/jdallen83/sm_content_clustering',
    author='Jeff Allen',
    author_email='jeff@integrityinstitute.org',
    license='MIT',
    packages=['sm_content_clustering'],
    install_requires=[
        'pandas',
        'fasttext',
    ],
    zip_safe=False
)
