from setuptools import setup, find_packages

setup(
    name='sentiment-mlops',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask==2.0.1',
        'werkzeug==2.0.3',
        'gunicorn',
        'pydantic',
        'sqlalchemy',
        'psycopg2',
        'redis-py',
        'torch',
        'transformers',
        'scikit-learn',
        'numpy',
        'pandas',
        'prometheus_client',
        'python-dotenv',
        'scipy>=1.7.0', 
        'google-cloud-storage',
        'google-cloud-monitoring',
        'google-cloud-logging'
    ]
)