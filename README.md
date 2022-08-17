# GIScience-CV-old-drawings

Abstract:

Landscape reconstructions and deep maps are two major approaches in cultural heritage studies. In general, they require the use of historical visual sources such as maps, graphic artworks, and photographs presenting areal scenes, from which one can extract spatial information. However, photographs, the most accurate and reliable source for scenery reconstruction, are available only from the second half of the 19th century onward. Thus, for earlier periods one can rely only on old artworks. Nevertheless, the accuracy and inclusiveness of old artworks are often questionable and must be verified carefully.In this paper, we use GIScience methods with computer-vision capabilities to interrogate old engravings and drawings as well as to develop a new approach for extracting spatial information from these scenic artworks. We have inspected four old depictions of Jerusalem and Tiberias (Israel) created between the 17th and 19th centuries. Using visibility analysis and a RANSAC algorithm we identified the locations of the artists when they drew the artworks and evaluated the accuracy of their final products. Finally, we re-projected 3D map digitized features onto the drawing canvases, thus embedding features not originally drawn. These were then identified, enabling potential extraction of the spatial information they may reflect.

Full publication: https://www.tandfonline.com/doi/full/10.1080/13658816.2021.1874957

Running the RANSAC process:
1. Download all files and place them under a single directory of you choice.
3. In case you have downloaded the files, rename the drawings as follows:
 - de-Bruyn: de-broyn-1698.tif
 - Tirion: Tirion-1732.tif
 - Henniker: NNL_Henniker.jpg
4. Update the paths in the 'process.py' file
5. Determine at the end of the file 'process.py' which image to be processed (0539, 0518, Henn, Broyn, Tirion or Laborde_b).
6. Run 'process.py''


The project was funded by the Israeli Scientific Foundation (ISF). Grant #1370/20 and DSRC (Data Science Research Center) at the University of Haifa
