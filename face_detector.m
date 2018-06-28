
% allImages = dir('D:\Bhatti\matlab\*.jpg');
%  detector= vision.CascadeObjectDetector;
%  detector.MergeThreshold=25;
%  for i = 1:length(allImages)
%     imageName = strcat('D:\Bhatti\matlab\',allImages(i).name);
%     image = imread(imageName);
%     image_size=imresize(image,0.4);
%     figure, imshow(image_size);
%     bbox=step(detector,image_size);
%     out=insertObjectAnnotation(image_size,'rectangle',bbox,'face');
%     figure , imshow(out);
%     var = imcrop(image_size,bbox);
%     figure , imshow(var);
%     saveImage = strcat('D:\Bhatti\matlab\saved\',allImages(i).name);
%     imwrite(var,saveImage);
% 
% end

image=imread('1.jpg');
subplot(2,2,1);
%figure;
imshow(image);
Facedetector= vision.CascadeObjectDetector;
Facedetector.MergeThreshold=25;
bbox=step(Facedetector,image);
out=insertObjectAnnotation(image,'rectangle',bbox,'Face');
%rectangle('Position',bbox,'LineWidth',3,'LineStyle','-','EdgeColor','r');
subplot(2,2,2);
%figure ;
imshow(out);
var = imcrop(out,bbox);
%figure ,imshow(var);
Eyedetector=vision.CascadeObjectDetector('EyePairBig');
Mouthdetector=vision.CascadeObjectDetector('Mouth');
Nosedetector=vision.CascadeObjectDetector('Nose');
Nosedetector.MergeThreshold = 25;
Eyedetector.MergeThreshold=15;
Mouthdetector.MergeThreshold=120;
BB=step(Eyedetector,var);
BB1=step(Mouthdetector,var);
BB2=step(Nosedetector,var);
% BB2=step(Nosedetector,var);
% eye=insertObjectAnnotation(var,'rectangle',BB,'Eye');
% mouth=insertObjectAnnotation(var,'rectangle',BB1,'Mouth');
% nose =insertObjectAnnotation(var,'rectangle',BB2,'Nose');
subplot(2,3,4);
%figure ;
imshow(var) , hold on;
rectangle('Position',BB,'LineWidth',2,'LineStyle','-','EdgeColor','r');
rectangle('Position',BB1,'LineWidth',2,'LineStyle','-','EdgeColor','r');
rectangle('Position',BB2,'LineWidth',2,'LineStyle','-','EdgeColor','r');
face = image(bbox(1,2):bbox(1,2)+bbox(1,4),bbox(1,1):bbox(1,1)+bbox(1,3));
ftrs = detectHarrisFeatures(face); %Plot facial features.
subplot(2,3,6);
imshow(face);hold on; plot(ftrs);
% imshow(eye); 
% hold on;
% %var1 = imcrop(eye,BB);
% imshow(mouth);
% hold on;
% imshow(nose);
% hold on;
% imshow(mouth);
%hold off;
%figure ,imshow(var1);

% % imwrite(var,'f1.jpg');

