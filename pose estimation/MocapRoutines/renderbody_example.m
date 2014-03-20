% either use something like this:
skeletonfile = 'D:\vision\GranadaDemo\MocapRoutines\02.asf';
motionfile = 'test2.amc';

% or something like this:
% skeletonfile = [];
% motionfile = 'tmp.bvh';

option.shape = 'exp'; 
option.render = 'surf';
option.style = 'flesh'; 

shade.FaceAlpha = 1; 
shade.FaceColor = [1;1;1];
shade.FaceColor = [0;0;0];
shade.EdgeColor = [0;0;0]; 

renderbody(motionfile,skeletonfile,option,shade,[],1,[]);

% also see help material for controling specularities, and other lighting
% and light options for advanced rendering
