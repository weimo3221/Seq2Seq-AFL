## STEP 1: Installation of the software environment

- Build a Python 3.9 environment with conda (or have a Python 3.9 as well)

```
$ conda create -n work python=3.9
$ conda activate work
```
- Install the relevant libraries (note the version of cuda used here, the version of cuda does not have to be uniform, but the versions of the other python libraries must be)

```
$ pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install pandas tqdm einops timm flask 
```
- Install afl-cov

~~~
$ apt-get install lcov
$ unzip afl-cov-master.zip
Then you can see an afl-cov-master folder()
~~~

- Install afl-fuzz

~~~
$ tar -xf afl-latest.tgz
Copy afl-fuzz.c code for llm testing to AFL
$ cp afl-fuzz-time-llm.c ./afl-2.52b/afl-fuzz.c
$ cd afl-2.52b
$ sudo make
$ sudo make install
(To install afl-gcc and afl-g++)
$ cp afl-fuzz-time-llm.c afl-fuzz.c
$ sudo make
~~~

## STEP 2: Testing models in 3 programs

**Please be sure to follow the steps below before each test**

### STEP 2.1: Configuring the server with llm

~~~
Go to ./CtoPython/PyDOC

1.Modify the contents of line 8 in module_app_model.py:

program-n = {program name}(The contents of the program name are included: nm、objdump、readelf)
For example: program-n = "nm"

2.Modify the contents of line 16 in module_app_model.py:

base_model = {Location of the folder for models}
For example: base_model = "models/Seq2Seq"

3.Having multiple graphics cards to test requires modifying the port numbers in module_app_model.py and module_client.py to avoid causing conflicts:

Ensure that “port = {port number}” in module_app_model.py line 106 is the same as “url = 'http://127.0.0.1:{port number}/'” in module_client.py line 15

4.Create a new session to run llm's server
$ screen -S {session name}
Go to ./CtoPython/PyDOC
$ python module_app_model.py
$ ctrl+a+d Quit the current session
~~~

### STEP 2.2: Installation of test programs (3 in total)

~~~
Installation is performed according to the programs selected in step 1 of step 2.1, and the installation commands for the eight programs are as follows:
****Note that even if you have already installed it, you need to remove it and reinstall it.****

1.Installation of objdump, nm and readelf

$ tar -xf binutils-2.27.tar.gz
$ cd binutils-2.27

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
~~~

### STEP 2.3: Fuzz programs and data processing (3 in total)

~~~
The fuzzy test and the data processing after the fuzzy test are performed according to the procedures selected in Steps 2.1 and 2.2, and the test commands for the eight procedures are as follows:

1.nm

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../nm_out ../binutils-2.27/binutils/nm-new -a @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../nm_out -e "../binutils-2.27/binutils/nm-new -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite


2.objdump

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../objdump_out ../binutils-2.27/binutils/objdump -x -a -d @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../objdump_out -e "../binutils-2.27/binutils/objdump -x -a -d AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite


3.readelf

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../readelf_out ../binutils-2.27/binutils/readelf -a @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../readelf_out -e "../binutils-2.27/binutils/readelf -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
~~~