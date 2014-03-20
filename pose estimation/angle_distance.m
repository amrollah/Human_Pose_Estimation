function D = angle_distance(a,b);

% distance measure in angle space between matrices a and b
%
% returns a naxnb matrix where a is dxna and b is dxnb
%
% option 'cosine': d(i,j) = 1-cos(ai-bj) 
%       (this makes it less sensitive to small changes in angle)
% option 'absdif': simply takes mod of diff between angles, accounting for
% wrapp-around effects. ie returns diff between 0 and pi.
% another option could be something like d(i,j) = sin(1/2|ai-bj|). 

%option = 'cosine';
option = 'absdif';

[d,na] = size(a);
[d,nb] = size(b);

a = (pi/180)*a;  % converting degrees to radians
b = (pi/180)*b;

if (option=='cosine'),
    ca = cos(a);
    cb = cos(b);
    sa = sin(a);
    sb = sin(b);
    D = d*ones(na,nb) - (ca'*cb + sa'*sb);
end
if (option=='absdif'),
    for i=1:nb,
        % calucating distance of all vectors in a from ith vector in b
        diff_a_bi = acos(cos(a - b(:,i)*ones(1,na)));
        D(:,i) = (sum(diff_a_bi))'/d;  
    end        
    D = (180/pi)*D;
end
