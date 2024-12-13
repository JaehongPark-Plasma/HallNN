function [y1] = NN68(x1)
%NN68 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:29.
% 
% [y1] = NN68(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;1.99951024573095;1.99936449657107;1.99886880123766;1.99830515183053;1.99891031114182;1.99871930539052;1.99891394704112;1.99970743670856;1.99807447911885;1.99826209148183;1.99973947762575;1.99911337506075];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.59088781942302470718;0.088043064635477299995;-0.27932034191815080515;-0.31923460052101676476;0.60728907786283525816;0.57093947211041873313;-1.3301304903757238396;0.14796978394614229702;-0.066555327997901120307;-0.096336929159083664609;-0.6258276094281367552;-0.26479393807072482137;0.2421945754591426081;0.22760905327269867482;0.016612910528285049216;0.99196693739415586766;0.5433617792537540625;-0.044996666633857977657;-0.24544340178187301627;-0.25037456197151491955];
IW1_1 = [0.24377296449186480753 0.57421049612465602241 -0.28643539870178119644 0.29044036036346565499 -0.050661580550033905035 -0.35722242409429844079 0.49157043410804168726 0.27132483457475625732 0.15889437504375147903 -0.62789232315116483285 -0.24229721000584741875 0.033416687411145000464 -0.050083464877842841234;0.25637161113453754702 0.14169887570153588685 0.37223024568871998508 -0.41693117993745337779 0.24878200534091948759 -0.083385366733274110529 -0.10319809738175586555 -0.11639648415708883644 0.067709884914151058366 -0.055704004439002544058 0.085667404703751545703 0.043622353031321683958 0.028491602803061229915;0.014493830043698745225 0.067910703073573719379 0.052977490263429856399 0.086262178090794228158 -0.015299900734957157591 0.20577504220270725899 -0.16136583798954581837 0.55315098962808739724 -0.12305493585487771446 0.30198530824480862567 0.078491826901316258391 0.05462428704873808355 0.011522505732642775167;0.59666173318114090751 0.26231226661552059509 -0.11668107959904990001 -0.25245844405528045096 -0.2097001659375155036 -0.091702292506120303139 0.12767553357555436189 -0.1466990199180280563 -0.13849228767302004806 0.010864154885462402184 -0.10037074688989248983 -0.069979081399048986833 -0.081141294316069395576;1.0041149437792507015 0.67001936973228803929 -0.60651243114530328793 0.040171104120227563161 -0.29416235796931694546 -0.40922994084125219327 0.18515658106642343683 1.2646803184634822781 -0.16735561277000579605 -0.087951075480283219532 0.15190277015821276874 -0.0042933150433104175572 -0.041150204401440244883;1.099218800803596352 0.62586808920571368464 0.30987523391829235875 -0.87232856525521385027 -0.4126702251335668814 -0.37859266406986008535 0.40825000195337551867 0.25513007277002014428 0.023367165164034777586 -0.26754735580246452598 0.22336265584749634416 0.03201652778941157379 -0.017845680112711103521;-0.10981429892057513464 -1.0244412803478388785 0.6192670674422496857 -0.71498343779669437659 0.68964813598095286995 0.54024179349365097913 -0.95921952245387598346 0.7499326185020671609 -0.056472215427426863954 0.66233014284828783946 -0.12013567138835790082 -0.0070187903027532153044 0.053168215054069559167;-0.87633908447026998001 0.092285225045852653514 0.029582568935997373416 0.64988651977378941726 0.086601951538480978487 -0.42668595224327410254 0.48685981725190813352 -0.45690135545367843717 0.32333909128058341276 0.00060225165617889750425 0.036003214483986099859 0.1350762157253589113 0.27727765935180442858;0.07735697722449140501 0.06741952306847430465 -0.32828397596242830936 0.4955625620871476622 -0.54269232347639795311 -0.37476724227822222346 0.63546144560401829349 0.26782568693904029722 0.067206594832144458951 -0.34089512144023337292 -0.010258300347676592087 0.034046467244316022671 -0.043474537574687770192;0.17320385220590125974 -0.47265440877014852639 0.43996189725192175191 -0.24895144207925895552 0.31242475102851197111 -0.21941811562097859878 0.30969231097037280831 0.18116692448558055495 0.034910881563785953197 -0.84080362170903888419 0.10149482668955603359 -0.025096376358577185406 0.0060022994793387258322;0.53342816802127379638 -0.53696788507787207578 -0.62911668780947938018 0.28154357860936035651 0.30704856213267073262 0.27107572710936311822 -0.21006495613741144046 0.70690276089342196641 -0.010493205500754953369 0.18201573224399852924 -0.25066401765367485321 0.0066276940572016287967 0.022187196413956080526;-1.102559201574025094 0.73879622044717674356 0.8860983757339104816 0.80126344987599251724 -0.31098076524861612002 -0.60622008486821477025 0.18731363094037725703 0.74407372133933680214 -0.19942155026593627598 -0.26466880140752913064 0.54391506749973228541 0.095200977213410434286 0.063498031210124067369;0.0067434665608676318954 -0.056855808489225639901 -0.19161340944010918363 -0.6584350705313116725 0.3223207817263564201 -0.39420025688444765288 -0.58320928338297084448 0.23442729326923625321 0.069161637151761504172 0.51896239862320470149 0.04023649446514253325 0.093476668332344040047 0.3978930880126477776;0.23228793708257888806 0.29351573731166474834 -0.49970999193030635199 0.03765582578103701672 -1.0821313860601262657 -0.22245959493742817337 1.1040563907220168893 -0.36955707047536012766 0.11918176653252038344 -0.76054147045149977924 -0.17926000769897523779 -0.095984013448308466709 0.12793573592482629375;-0.41476896936807433613 0.17015209361973876456 0.4236412537987943594 0.10725596420534994069 0.15713570719678640608 0.57772669448443370044 0.49005336093497509076 -0.045507975311845486055 0.5691101698580360857 0.10605018457053669301 0.63915678927645092067 0.034621939012733235486 -0.046167974954648034747;2.061977896817311251 0.43451785455008395953 -1.7329187883125805048 0.74593685003769716335 0.21277020009359959296 -0.13813847989175592756 0.21797609687545743684 -0.99262366477812236099 0.038547791369235216707 -0.57489935625309029898 0.91221261605006898066 0.098185221747752904342 0.010157718836308778781;0.020939388140879703487 0.76534436923647064521 0.5161699615369821359 -0.042405354377451610903 -0.27673573246518851088 -0.28975731382620689702 0.016969791662273726873 1.2773922768568588548 -0.22657556226716907277 -0.14678014078919904284 0.24031994000141673751 0.0068953328737485410777 -0.065308142871178698208;-0.44960931478561527719 0.043885611382727035545 0.23062516746006037072 -0.26355347016048763509 -0.2677474260462250899 -0.10310230223104559222 0.56365723058342487306 -0.10218437422147770566 -0.12481492073129382692 -0.0091759380046469614378 0.15662808162797187128 0.072765389009773764895 -0.04133394194504733804;0.42241033229386232284 0.50576418047578486359 0.20215046952057397212 -0.38188602446573910454 0.050795948201109650255 -0.23338380202396308283 0.45936453981169056426 -0.76335255015372427412 0.1562507672214826504 -0.86953275807330865366 0.16265471138344886337 0.060994224000485861015 0.07458177830796546004;0.030514127807629447958 0.16513292680979846372 0.065226247356073310812 0.10670857145493643781 -0.43468758775888460821 -0.040128340451932795196 0.60930086211755107861 -0.45456774392425003395 -0.070012970195698312481 -0.18846399196613281002 0.060223935248083858285 -0.0083751115636548079113 -0.031603753841018725201];

% Layer 2
b2 = [-0.69099109813704828742;0.31578199790914801559;-0.6258323465099521199;0.38746678618750124423;0.34461502355018702737;0.19365726834823768221;-0.23213041833657624413;-0.10927121352542291943];
LW2_1 = [0.15222785953718148688 -0.43186053659812340122 -0.12857090745773760987 -0.30132487975646332634 0.0051960385968070083723 -0.094463038777078289776 -0.042281922926408928298 -0.053611752524324866875 -0.34042383775049267491 -0.054649556338324190019 -0.24947259884884698256 -0.09514068761046391709 0.09131018688528244609 -0.045516802979962377174 0.040042195700061240793 -0.0027966727558706031345 0.12465366142551621842 0.13251951557676006077 0.0089905963998317406372 -0.21902968510391493395;-0.10690894876679166159 0.21874466491491217912 0.039331934315281216752 -0.30504210462681674576 0.36728873834656811104 1.4637304122462624356 -0.038060996939197122568 0.2499528202937642829 -0.73557193806017540094 0.020791369295630392211 -0.41397904949391961527 -1.055120147964994759 -0.16594812773239195347 0.41080877120716324802 -0.44838671361322257614 0.28452724626361602933 0.48810364528803229511 0.71613095107952751128 -0.62404537707416929404 -0.10585955737951989464;-0.23706102878600193717 -0.011881949585290658977 0.2508766695610741837 0.072242162911606705444 -0.78589705300838674784 0.52056731464925032338 0.55017782502873091577 -0.38983813684484031059 -0.27121698099414781602 0.80388521125232936804 -0.87215877167903410605 0.76860282922915246218 0.13916413957282250391 -0.017021556391773110667 0.32769108676192837404 -0.75926981338517585129 -0.36710301577884379975 0.36937625441875471255 0.16804797764992771003 -0.27562041188838160011;-0.32884598411428150877 0.094124431270670272376 0.53855670853295123379 0.22565247871133584101 0.5477924528560961237 -0.18411820997981623305 -0.93102415424373596853 0.16726049416426980176 0.19108440051764308265 -0.23049313943224350454 -0.086159523144321667476 -0.6617264333454638825 -0.18973165024631344355 -0.077429723741727560227 -0.2842935334993661467 0.64379671327804210978 0.54192054221135788072 -0.36531584855471849194 -0.088160154941011537044 0.14003649111266952554;-0.077286884896057714278 -0.19735628372472394099 -0.096995238956933529928 0.093356830805906315662 1.1652185691514234112 0.084281389824478289419 -0.2956236255467509233 0.42444709896066995514 -0.015758423411878087761 -0.62213109067219751758 0.073848123338087018341 -0.96004313864258905564 -0.20796969343425991683 0.18216167671386762628 -0.57910163430045680233 1.0470372206618092648 0.057851946790438167623 0.19234621740282040192 -0.30658402144314828019 0.35866457327558659562;0.076624336835252479516 0.83278470234291623875 -0.052748736260528963915 -0.22105442941215228125 -0.50124328988318778144 0.13597065634812668944 0.65587854059387373784 -0.68591427240505142837 -0.24163823163058387178 0.85982853320867480562 -0.43805073885891060437 1.1554037199516293821 0.34399284039506383692 0.067474166647268338237 0.63665993379658014639 -1.4224167457566714301 -1.0321679716987746112 0.20078702570792431037 -0.0005262156897664852645 0.065221866422342289482;0.79550771465319269904 -0.34494949571037064606 -0.18794386770197013536 0.0018984492817681756019 0.15638259934029172249 0.17013482631915594068 0.14638449034259098935 0.23291748763747929796 0.15118733145125851092 -0.34834366687186452127 0.27049937780043942226 0.2129711657854276019 -0.35994070073438499957 0.70001210700427130007 -0.15460221057390888255 0.33622723471997884026 -0.052579988048360983 0.56707224759695773475 -0.29368603934367887298 -0.13258244663979409927;-0.15051265148920256465 -0.095011032893114660824 -0.34994518538559848952 -0.26340891132885024817 0.088564434976252465193 -0.15290659445126583704 0.13347008644327765237 -0.051104860877568922262 0.013468577151237679304 0.013472702739092476051 -0.21727427422283415726 -0.081413409568142636186 0.039956762752068775146 -0.071322211540584268263 -0.030857897945474402346 0.023990240168965681206 0.034502358142339130398 0.10384736400583863269 -0.11834246335602727207 -0.14674301924540389397];

% Layer 3
b3 = [-0.72411432363724714811;-0.87126800779250401785];
LW3_2 = [-0.94965873714568027619 -1.4222629476705332952 -1.1008377084692255732 0.94504649836478327174 1.2705357580494966996 0.99066801615645949664 -0.3492135136067847645 -0.77449149604553235715;-1.1655999902014468894 -1.4143814013593156709 -1.2200237811662051346 1.0973688029638271058 1.3369425780464463394 1.1863459286066415643 -0.26239619144768211445 -0.27980729099205481347];

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