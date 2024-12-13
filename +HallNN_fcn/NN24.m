function [y1] = NN24(x1)
%NN24 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:27.
% 
% [y1] = NN24(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99863902222647;1.99926385962227;1.99896684695323;2;1.99834269350155;1.9985843912067;2;1.99845908253947;1.99881166696164;1.99822131745036;1.99809729666903;1.99928142009474;1.99929797949473];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.27129443505941441828;-0.32250106890320534569;0.30786692262055426195;-0.38258739362424909469;0.10335286293579472316;-0.11582682270013297532;0.034141288986344893452;0.44285297377339372105;-0.16965112540081531134;1.5533315505972027371;-0.40493091719576490561;-0.26869564273371948637;-0.24533618535700069119;0.50664669156390318427;-1.1314353231607474637;0.040978705628731466881;-0.39376501138462877138;-0.56265510676743968332;0.17467949722192635531;0.074454484083357766466];
IW1_1 = [-0.29205359274710196438 -0.10848143271532352527 -0.13133347956176885662 -0.30350531415853498896 0.16124034045232804591 0.30301318704883944966 -0.12634201662945462874 -0.88856282242922379755 0.017738953242044976089 -0.4637215077865934898 -0.37589943105608597529 0.091981710366691532554 0.058027385619185294086;0.24422182915722207008 0.20052918323060342498 0.039766162117943440468 -0.31633610320694460771 -0.026057269826080114994 -0.1315272064064925972 -0.15054673082978040077 -0.48570405457894799683 0.044518955422182655834 0.054549981399908631374 0.11580650969080888424 0.030276576031866474226 0.0074379134506996772216;-0.61743657761578596421 -0.48609877013460878192 1.0424049291355346192 -0.4069546746033354534 -0.0013833694385765676026 0.4908230756924898408 -0.17753487022257127825 -1.7706651174522747283 0.051046773465358616373 0.43476363056462458134 -0.27454493646084437986 -0.010480743935422726087 0.054909918609344757212;-0.34207118637582500931 -0.06528608837516572394 0.61399357608859450597 -0.18181938564287958293 0.22152721919940893125 -0.14759447999533731322 -0.029171291821805146738 0.018008046084092663164 -0.042541235040186364813 -0.42148820142319737281 -0.1342326760829114507 -0.055001511852412671155 0.015644764990101981544;0.11569757378856111718 -0.16964513690328217321 -0.43851770073039864073 0.073902677033648736526 0.057942853226764644281 0.23463921661721826029 -0.0885947443374762722 -0.049487832541256569774 0.071550739432860360756 -0.19086754911786663058 0.20453063516039135528 0.02612912709829162336 0.11673205319931351243;0.047624985850774945173 0.40342367317185345854 -0.006675125268445907499 0.081055781124230147361 -0.59590309010262354494 -0.3029730139504623887 0.7757176789828701402 0.27020046778465800497 0.015956403194816833752 -0.35016370637727389781 0.093325575373369815124 0.043588510599131886913 -0.050710074975385585616;0.38969761761278598478 0.10843944358937299133 -0.46700566034033852425 0.43867853302096637691 0.6449509430523675535 0.044983483026725321063 -0.64004066725254071812 -0.089323605362793156992 -0.014456509061583413048 0.024527358516370348362 -0.0031851956688309090993 0.036835041292612356878 -0.024484412706329592574;-0.14878435065795969638 0.8302273805470187229 -0.56478301837487254744 0.69035940523675998204 -0.43809991180528995436 0.15573904360354171028 -0.055606891230190441666 -0.80376452684971544382 0.02030625094213042714 0.46229324476287836365 0.18731437974190301432 -0.035431595055103619341 -0.067574486306790193524;-0.19054031308017729818 0.2073649891393245237 0.037066130153663362501 -0.20637495731013197342 0.10085275899094539587 -0.20961343130109863497 0.036669337393834253169 -0.0027554350021042738755 -0.017160843344087611112 0.053340406181247385087 0.070650508452737598164 -0.011735826256324135078 -0.043830061353184214534;-0.14967729670146015519 -1.1955539763801676933 0.0096031805716927966243 0.10743438774169636329 0.64631462987207188853 0.22720687226956762239 -0.14283921108467442784 -0.31996311880925504267 0.011486707283501389545 -0.75134038285561954851 0.17195547596970711668 -0.16501505346785527717 0.023590641614025631423;0.23011191371457317967 0.58347599747856426777 -0.30282544549254097888 -0.30229107996070564424 -0.96095644497996579592 -0.25083067151436477804 1.2229490369639468561 -0.65427405207684596888 0.11609847022738579991 -0.50462181969746089383 -0.14853820665743222573 -0.24143597573504385623 -0.10649777681629155057;-0.12189763007164074182 -0.081262062598433729144 0.23263458142023654696 -0.017544949770508997539 0.15582381098329642355 -0.072109882840620836975 0.013342738354685971386 0.060881920765753896885 -0.032158571270552813171 -0.15440268945320193006 -0.058726959693804918483 -0.15624231361916110505 0.035068032077244501443;-1.7975948945447692839 -0.38418263479575853747 0.050194614692476755002 2.0332506246549590934 0.18439179965150037654 0.1258695272818909916 -0.24719925024784916712 0.25161390817451911861 0.026069566067834188533 0.17378284562213724973 0.21535941423830295971 0.045801701075650319561 0.07975000074910121306;0.15438175266517276119 0.099819648497034021983 0.0043274223702078002188 0.49313486737183787056 0.19673304946707903529 -0.27347671723498556684 0.069583333098563182606 -0.45980366652196968946 0.63978863476751424155 0.13740261579135951142 0.21970153358930613607 0.044093397770469687502 0.090060109071277247295;0.53601547102201507755 -0.81467799149662989411 0.45429234558079323314 -1.1548507723726164631 0.70992178829289642739 0.4378329624048530877 -0.99450296744665500626 0.7530061116510903263 -0.031971732470949594529 0.73012715347562784363 0.22650076400289395617 -0.089957970751254751063 0.1792994911538413616;0.20385794425417258502 -0.54218838425222370958 -0.74167078344289971703 0.96509909510271840727 0.13834498566330105662 -0.052780218190929810429 -0.35354620990817470227 -0.17314842559049004045 -0.25877230302235249004 0.028330324255483908813 -0.25276146710429142628 -0.089941660771547471409 0.011602805889640457546;0.098007559487458734138 -1.2270624552169870913 -0.49117649684839892466 -0.078854980845060138051 0.53161830326814452086 0.62289847035158241173 -0.17253355765796055121 -0.59908181166781260263 0.03083998621468319809 -0.028005949431931319304 -0.1615176673286051745 0.0099134370175361207628 0.076322152867810938437;-0.82695204794255439129 -0.5016490381760436712 0.46445771840511823614 0.72481548709760312832 0.070539538542541982413 0.83949241683304620665 -0.46476600783151789109 0.5455946180551706215 0.0015561444880084263943 0.57442261839919206423 -0.13274346524683530713 -0.16546736411710069592 -0.2754938412966986272;0.39757106864699587323 0.40626903713203438295 -0.37466148817710931418 0.28541327500782381899 -0.42513827552234634277 0.056399614088003029166 0.70512953537913158897 -0.56913290622058543278 0.066474168056862792398 0.34709114325264600609 0.032985785829352076703 0.0954239133980588905 0.058916910274231869271;-0.021797763093945446422 -0.035814859751025399093 -0.13837067455868592991 -0.25910652917849841259 -0.13792387154297405805 -0.37107955486331217765 0.22574151031561093261 -1.3001012416988257936 0.19539558982191823633 -0.3609163159124788578 0.12098777525669465738 -0.030704284046787294615 -0.088595880424678655207];

% Layer 2
b2 = [-0.21551383159385364974;-0.262984098804208144;0.5117607991314674365;-0.093400210619688686653;0.078003530971766946167;-0.013644493459887562653;0.26575297019700983014;0.11295706984474453149];
LW2_1 = [0.18538607688468022872 0.032001915514215398262 0.28419335153934666138 0.18729323732755154275 0.18110735530321672715 -0.25184226071559345783 -0.024368108527601690938 -0.01235736231973111332 0.2109312909609036002 -0.01115169007100707925 0.29088778662738506986 0.043109141092979939847 0.16605499600108031277 0.059276656001524624662 -0.3013639544217383559 -0.1969148093207535033 -0.24533061973141809253 0.0091529140313914326976 0.076563414403669843855 0.037463155299125076858;-0.076259299375210046201 0.02023598860126002083 0.034568784121409859567 -0.22932640757880415339 -0.21635327097526835249 0.13634349821650815215 0.04624830966882092359 -0.27338751078846784681 -0.35071828651371089736 -0.11806764786956412683 -0.024718068152799886616 0.17979802845017975321 0.013536678866628718346 0.1154079698859694636 0.041640127319498058955 0.037185963943517655328 -0.24832119419043879871 0.18625317013543490896 -0.028606736823433641381 -0.038992669264296245246;-0.75779273316294104568 -0.10971806069879749823 -2.0353271892427446943 -0.85541302499530891268 0.45310148918837478416 -0.83657900661425366184 0.68037270587625553464 1.6295999945024908229 -1.0273217558199880806 1.36965065873801084 0.96478637308581327581 -0.42699641518083553127 -1.7532062608675533788 0.73892795018584855882 -1.6705819482973229828 1.7935208533146926868 1.4053980752904442486 -0.87745308435368374322 0.98170643768818910413 -0.94314040810048860308;-0.044572179875952099015 0.033479695184834235011 -0.031549551599694941606 -0.0068114492872662054171 0.18046022298666256067 0.30826893064485977325 0.32161381120897031272 0.075150764700752173408 0.05005970261766925361 -0.0226352664885039849 -0.018120010134402529667 0.1641650182752870113 -0.16377706302195590982 -0.035766551126911197422 -0.059618554530211191611 0.027961068551673462501 -0.063986566812862077924 0.028406598147538771049 0.13967782000912101759 -0.06990636416884692117;-0.14654335749564770253 -0.19564716011854288436 -0.33489682884739824242 0.48655201872202236135 0.49652558631932630639 0.75116945233125587134 -0.57586984156693232784 -0.26083417362546779117 -0.3941101669005353525 -0.16064891131650005374 0.79264385150397254076 -0.025474071782180880003 -0.041243595443402489287 0.11470187370596038323 -0.81041360452519140001 -0.30851684762562658904 -0.13633277256357237417 -0.17949077982527938957 -0.0050751543667642501703 -0.19163842209988782295;0.057839940046560088527 -0.39608313752025314791 0.032994323192823617197 0.15901395344814236998 0.11247514013433465574 -0.54284008094942881595 -0.43008222850461602649 -0.050126609725628802627 0.44752300291734858906 -0.095815061432190207213 -0.27394957864761165789 -0.22224576869266407519 0.052639185178640905682 0.07182428486402524237 0.0053791557714524913456 0.038890153787731196466 -0.38673357387336371183 0.12994238277498840195 -0.14032351464384495987 0.039227584422197252223;-0.27205453581355837356 0.11541091268868582109 0.13314563298773643418 0.12905979464794459233 0.055609897387342337172 0.18077528483795879399 0.27376720564356415988 -0.084846167314425152695 -0.079205705910728757546 -0.1793368438274490495 -0.36154488455001404512 0.30826071150041317592 -0.17837791573564948377 -0.087670033392864429866 -0.17672125160260160337 -0.035206145031324881378 -0.66452769810312539711 0.15535748821765005268 0.2285123502441943899 0.24871114144071551633;0.27892329958723277583 0.090807338057670480835 0.20425774360151141562 -0.059135470494140016107 -0.0055964024485458925898 0.054912117361026975548 -0.1321606523172560721 0.25690978178522982933 -0.31060710702957577967 0.17577353593912131702 -0.25276157231017459814 0.087224722994730816139 0.24112411584726450853 -0.040485478729196631986 0.30426015900001152081 -0.035466079385537536972 0.46849091868992820409 -0.33248468808024916887 -0.11996698675392056255 0.028615814743109761059];

% Layer 3
b3 = [-0.80522348198012372311;-0.81844954056536722842];
LW3_2 = [-0.16951725650417590052 1.019038121526730345 1.3147318524558526676 0.57197457973079390836 -0.1250960349321364462 -0.8786796264112478605 0.31346663609700414765 0.36561376762549341324;-0.30559268553029106386 0.81835358528369184228 1.3004488183888984754 -0.2105156075792163628 -0.099508252188388615633 -1.0879088327584434115 0.43663731814981970869 0.21287083105646739667];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [2;2];
y1_step1.xoffset = [0;0];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = tansig_apply(repmat(b2,1,Q) + LW2_1*a1);

% Layer 3
a3 = repmat(b3,1,Q) + LW3_2*a2;

% Output 1
y1 = mapminmax_reverse(a3,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end