# A simple image retrieval demo with pytorch and flask

This is a simple demo of image retrieval based on pretrained CNN.

## Demo.

The demo video is shown downside.
![image](https://github.com/SongKaixiang/image_retrieval_platform/blob/master/retrieval/demo.gif)

## Usage.

Please install requirements.txt first:

```
$ pip install requirements.txt
```

Get the pretrained CNN model from [this link](https://drive.google.com/open?id=1TG_Fq_UryffsmV045u4MJGaWB-MJqNgI)
and put the model in path "./retrieval/models/".

run the following command:

```
$ python image_retrieval_cnn.py
```

You can vist "http://XXX.XXX.XXX.XXX:8080/" for the website, where "XXX.XXX.XXX.XXX" is your local ip, type "ifconfig" in command widow to get it.

# Only Test.

If you only want to test the retrieval proccess, just read the code image_retrieval_cnn.py for reference, and run the following command:

```
$ cd retieval/
$ python retrieval.py
```

The sorted images will be printed.
