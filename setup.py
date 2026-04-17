from pathlib import Path

from setuptools import find_packages, setup

FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / "README.md").read_text(encoding="utf-8")

REQUIREMENTS = [
    line.strip()
    for line in (PARENT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

EXTRAS_REQUIRE = {
    "nvidia-fa2": ["flash-attn>=2.7.3"],
}

exec(open("gigacheck/version.py").read())
setup(
    name="gigacheck",
    version=__version__,  # noqa: F821
    python_requires=">=3.11",
    description="GigaCheck: Detecting LLM-generated Content",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    author="Layer Team, SberAI",
    author_email="",
    packages=find_packages(include=["gigacheck", "gigacheck.*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Attribution-ShareAlike 4.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, ML, DL, AI, transformer",
)
