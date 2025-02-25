

- make sure the dependecies are present in `third_party`-- if not, run
```
cd <path/to/orange>
git submodule init
git submodule update
``` 

- compile flatbuffers
```
cd <path/to/orange/>
cd third_party/flatbuffers
cmake -G "Unix Makefiles"
make flatc
```

- build the schema (here, I'm building `objpose.fbs`)
```
cd <path/to/orange>
cd schema
../third_party/flatbuffers/flatc --gen-mutable --schema objpose.fbs --cpp
mv objpose_generated.h ../src/
```