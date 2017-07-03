# MAC
Tensorflow를 macOS/OSX에서 설치하는 방법에 대한 설명

* 참고 사이트
: https://www.tensorflow.org/install/install_mac

## PIP
$ sudo easy_install pip

## virtualenv
$ sudo pip install virtualenv
$ virtualenv --system-site-packages ~/tensorflow

## tensorflow
$ source ~/tensorflow/bin/activate
$ pip install --upgrade tensorflow


* test 하기
$ python
Python 2.7.10 (default, Feb  7 2017, 00:08:15)
[GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.34)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf

## pycharm-community
$ wget https://download-cf.jetbrains.com/python/pycharm-community-2017.1.4.dmg
or type it into the browser

*important* to configure the interpreter !
