# cog-tf-pix2pix
This is an implementation of Tensorflow's [pix2pix](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

pix2pix is a general purpose conditional adversartial network for image-to-image translation prediction.

Here are some example outputs from a model trained on a 455 pair dataset of Albedo>Normal cc0 textures:

Example 1 -
![output1](https://bafybeic7edpra52suajwpoczeosjz53oammrvfp5ic6qz4fl6swhj7jkhy.ipfs.w3s.link/image_at_epoch_0763.png)
Example 2 -
![output2](https://bafybeidt4c4rvw3u6qf5tjyl663elxygjsg7e7ghyk725yry2fgqgucswa.ipfs.w3s.link/image_at_epoch_0762.png)

Models were trained using CC0 data from [ambientCG](https://ambientcg.com/list?type=Atlas,Decal,Material), [PolyHaven](https://polyhaven.com/textures), and [Pixar RenderMan](https://renderman.pixar.com/category/117-texture).

Individual models are avaliable on [replicate](https://replicate.com/tommoore515), or you can try out the Unity based AI Material Designer Tool on [Monaverse](https://docs.monaverse.com/get-started)
