function [y1] = NN54(x1)
%NN54 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:28.
% 
% [y1] = NN54(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;1.99930219016442;2;1.99819547687575;1.99806809458557;1.99885411595864;1.99867054634307;1.99889538458961;1.99959140236946;1.99921479366699;1.99830895923486;1.99895014156362;1.99871860624491];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.34914142077945786724;0.76333100128612296764;0.07978574425905057832;0.70803201429232986985;0.094760826834922345929;0.34954210586524153692;-0.69554858905747551923;0.33911137881727865251;-0.7808807489912414912;-0.67527468463388995445;0.19620998522175206458;0.14917113414864432985;-0.32830775644654280887;0.15994275134589541754;0.032808422204565708402;0.39417168564045051982;0.20973926003372864169;0.37949791155448120428;0.41236637430378964009;-0.29875192553174506616];
IW1_1 = [0.12012376548205612459 -0.054555911766443511424 -0.074893744896651126552 0.039435024337130114569 -0.3238321769925248117 0.01279867119629327088 0.69616524728693396362 -0.062344168308087415109 0.063016208618984409506 -0.36471228365463514987 -0.27845439337673177915 -0.029066264995844651176 0.03592303273972492772;-0.042907562002956263492 -0.50055382347929044951 -0.10317812829992661916 -0.15313020102353325225 0.46727980765241278194 0.45876116882201450897 -0.44807212947669927683 -0.95824690882699348204 0.069392352093650994105 0.27131843544936318224 0.060068809274214256255 -0.08345713984466736346 0.11773384980514983722;0.23972664495408663177 -0.19101441554812076395 0.10240187230153188769 0.10277241210012189043 -0.12530464292181878183 -0.062664191555117598753 -0.41337655166059017464 0.34579437799520579278 -0.027733439214443215276 0.34867973608106705985 0.20022661922433637738 0.01426486334459944709 -0.098318318235352503143;1.0436732844669787124 0.29221380160173099361 -0.54119113836243193383 -0.4486918338991889299 -0.30036281856553226355 -0.50543273419129919333 0.081229347040164143201 -0.63775066263496127394 -0.032897874328084532325 -0.38272960470325356752 -0.040634479718577802554 0.13539315930525575027 0.028452005704446326623;0.030852049577563149529 -0.13098583471274849455 -0.22495247645354038424 -1.3265698755152548483 -0.20290111221469978031 0.062398815856265811564 -0.001596024856310085914 -0.16684007035008738273 -0.12223074077076673261 0.21446231256087261441 0.10224132247324260736 -0.18237591388764351485 0.076730464929106623195;-0.3266042865988513233 -0.19287877191020649792 0.18671515044459408017 -0.15479536088113760028 0.21783133258629405749 -0.029949822522329940661 -0.24948241037328192093 0.13063553676099726175 -0.14530273721700495693 -0.20722596288499586792 -0.39244072705793903166 0.035660105629871000077 -0.010798881172681972646;-1.5290682209635808597 -0.27023411207636238718 0.76351229604677428586 0.18437221225760255794 -0.28881566658289953287 0.10116177032037776407 -0.34005904114975199493 0.57185727184117107758 -0.028854518267662106895 0.4335997454305742238 -0.76818459738292055317 -0.18651870683038507259 0.0065232691795880683172;1.2445286162796731499 0.211488775715863081 -0.38520117862353309146 -0.08646761813929416951 0.37930622937479668844 0.00071793177472038050696 -0.39644165667173691947 -0.38446929748593272613 -0.063433105336604705959 -0.30602291452330032673 0.54998874872874514708 0.14703545248372132392 -0.0032519871198963511139;-0.21726103973027252603 -0.80960627137193086345 0.10469751154257467274 -0.055477588060492417921 0.52196709442739708251 -0.046714443507765004981 -0.018763316230602903462 0.82162985667025201231 -0.02442641364178756852 -0.53280590430240326594 -0.054270627691892864286 -0.042509656152969886589 -0.051773439983882327264;-1.0427988750444794341 -0.98879919806354188427 -0.25551428220709926364 0.29916832224993045175 0.64271075019927659255 0.53573811384269320879 -0.56404789163291946252 -0.25799699362717659623 -0.12020803361277322685 0.29517721565614263302 -0.60087486149443791739 -0.10105509325775016705 0.077487124295537293528;1.3324010550188329915 -0.29477795641330500898 -1.0518545390649727267 -0.19354108201302466785 0.13488816125355806985 0.21615388304414784315 -0.20298866509235763411 -0.030914599553170260537 0.020797890693681563012 0.1949654772729154939 -0.52427206433124640128 -0.099692410528551739235 0.035718978044722057841;0.47210463295675098028 0.74403783880089424674 0.022700733210575842203 -0.35634947627421265093 0.2327024628150876906 -0.14997845176504379094 0.16944543854746138556 -0.022975922889685224115 0.34411143847627312375 -0.14903501885909081759 1.2368657521458708004 0.10088350187040627526 0.035025590774840936747;-0.10869553332961628955 0.43379990016542724351 0.23245353512532263007 -0.69115628361165570848 -0.22260538014991879119 0.04603633180522807844 0.21439772845806603918 0.2838782985278339166 0.18589061498143821805 -0.079911264785977598191 0.33637378428515773976 0.034608515809850221023 0.037872440636082631282;-0.62587994833114346527 -0.46238385148032851157 0.44645069590164970785 -0.16435067336367148449 0.17020641232261088382 0.017519589829089105004 0.22807413629053041593 -1.3071332005639537677 0.099813353216845632221 -0.64086843973907392513 -0.36134000115004688558 -0.047211852386744523891 0.04647304713716987612;1.0915488301093538848 0.44696808896100886921 -0.50262763754740036326 -0.054919813973281120245 -0.14704250611777580637 -0.058849065590481530741 0.064294504817194883528 -0.42132697385300132975 -0.049986278094409156147 -0.19172039203918894712 -0.41808798608046893186 0.05684816160502650928 0.065291753811780942063;0.49831685656580609889 1.0159695358645599583 -1.341998161494324826 1.1178231386268855374 -0.80264441722314983085 -0.60498054354941599442 1.3283654175769643579 -0.73658147266904749895 0.019003511786361219904 -0.82874095340938835896 0.85859408781788904008 -0.075274963172658956867 -0.18940352831140219725;-0.25269849804054683728 0.089180120453099101518 -0.74859828892776525411 0.0050570398673308722057 -0.12187659899403364971 0.31133239900952985879 -0.23259746470308251265 -0.65375420912719550692 0.0020198636882844071133 0.15683484118181592604 0.60569434351005613237 -0.021342141242582757937 0.081116610026719479509;0.32505942607068166739 -0.50239381100706581584 0.16135554646612892959 -0.17714075288396263774 0.33399873714644606748 0.34285886639935447828 -0.69790589374124534228 -1.7079808542530647841 0.023786116814334345992 0.59501131458199563173 -0.12698633910490589316 0.10243336598519704739 0.012884930634074510947;0.4553277063626393617 0.11051088998668626151 0.073001834021002104391 0.040981281718105971867 0.059909091075244967628 -0.21334614482812597203 0.063974970169338077342 0.14864621709207892497 -0.02216849920507274474 -0.24741823713554422781 0.12853104527787387323 0.046905511538203714661 -0.034312651963358822838;-0.031564478097796035139 -0.47794839534962096916 0.021851450589894391707 -0.62399661118431304363 -0.02893971769210167827 0.23675327099165729972 0.13634208308163253687 0.41732886013364617339 -0.61803570728936030321 0.17779422940694625255 -0.53306314644244690015 -0.2935988602420161464 0.029554925607408789184];

% Layer 2
b2 = [-0.21838434935797301772;0.41392692958004095871;-0.32390193403959621765;-0.054308579869090295444;0.53374551554360960548;-0.10600180305683933046;0.66083307767755761208;0.24075821923304335792];
LW2_1 = [0.37471343736703055516 -0.0538954770180676071 -0.46237373599258485557 -0.25152129156469538174 -0.5501108631469654453 0.17024006834574401115 0.88946548541517178599 -0.83795738470241043494 -0.11969704089184048101 0.30410643497979805705 -0.53483737921761143053 -0.74281379204993702725 -0.080383922164115093056 0.24215515800513234823 0.49861560635253465712 -0.95834556613595933694 -0.043543846622385244671 -0.17843123806923891217 0.14471473816729316098 0.020606876188768937647;0.40198157992458211085 0.002835686052594774912 -0.25919866796620127936 0.15307304564314577755 0.041093963757788759539 0.062442865772054025053 -0.18072364843155258596 -0.31757197361964289994 -0.22554766789170205277 -0.45125766014073587229 -0.26374275022733523199 -0.051031931848287692688 0.12079593619207304456 0.049938162028981492035 -0.608879217241204862 -0.15505051849520382068 0.16262638916216382023 0.17188280336272368509 -0.0037609393488717045642 0.26462231716078638;0.31361655154994805272 0.08622910689859852873 -0.26365798946883339804 -0.73588180357405907905 0.19679371848647755905 0.65799310286759904454 1.0078118199392698262 -0.82353696286123723169 0.23472933255444897171 -0.50307594841096203719 -1.1286249177144558331 -0.49162043030461050064 0.43369606864529086643 1.2678703094953189812 0.87676359060467035089 -2.0320138256467190629 -0.86476880520486643089 0.47189806163620023893 -0.26454726183171145548 0.44985464024671234506;-0.18145580763451998885 0.31190141915818603291 0.26972542905349405906 0.26263072896625699881 -0.13126394874541300051 0.49221093013016209428 -0.13650408103531269655 -0.24224677798191360245 0.12366236768137878199 0.36440598627050740932 0.12753946164793555917 -0.22612082825900670091 0.055408493720620524692 -0.056806763462981003932 -0.19709958679090330835 0.23392913636216208784 -0.016618992564376624393 -0.15388374604133356338 -0.36179364169875832813 -0.23672592735220587823;-0.12454425326931997664 -0.1112826711186171158 0.68353694484439786461 -0.45599693802693069822 -0.15552215534582475298 0.12832774435990676465 -0.41785433025482648928 0.43865748748889399344 -0.062826260010631337916 0.72825106260831728999 0.68970064683547183115 -0.60667356710085063032 -0.53427078000414651715 0.084478347697470354549 0.41703191937999123384 -0.77535954947295460116 -0.50772851352412606829 0.067458151255790368972 0.3777746343155281572 -0.65567487815028768772;-0.57216022730718363309 0.0067937753637831149039 -0.24108340059948171752 -0.16926992681944683095 -0.031190445031851754543 0.21257467112783651197 0.073310761188281767242 0.22024115989313741704 0.029907252337928409702 0.11381682678982778245 0.021904448581116814365 0.024562540435269362982 0.1146673663844966079 -0.064836246015629955819 0.10757911454830561271 0.15616291256372946683 0.016210726599392929342 -0.062881963316778505257 -0.44616086992377679277 -0.11845965673868230628;0.25264906183886443047 1.0657198059236085985 -0.1459112894131917304 0.80962826370398222053 -0.53304341690760259631 0.31193547034815111152 -0.94849328201462490551 -0.12298954441745052479 -1.0825135734494617523 0.74785963750132655736 1.3312333482769156401 -0.820455794030878649 -0.9514995052405469389 -1.2556358166185486169 -0.28574559638477570189 1.0703786386867115699 0.62639120590838670566 -1.2936501928604529077 0.17375206901462997289 -0.59846368063020038974;-0.1232947310394988949 -0.17856360876271010207 -0.10790509342286110084 -0.24957090027964304313 -0.3365103228711846528 -0.054176101994635933456 0.15658385819923878457 0.97370859324603797891 0.15702566768468503855 0.46444365014420602167 0.91221937585507240076 0.015682621391569093877 -0.30961202649783520213 -0.48846927937961326371 -0.26622733644769264183 -0.037109533215171720821 0.10305779801910734672 0.46825492759656395236 -0.008099515661980756509 -0.38055440642515414584];

% Layer 3
b3 = [-0.64508618031686604244;-0.4399069579786996087];
LW3_2 = [0.83345426780216247398 -0.73469339898280183743 -0.77946148693939831809 -0.63937299106086709077 -0.085659643251049036317 -0.5321629116328128184 1.2464424719341204995 0.15102580641611859646;0.91171621555309667606 -0.64108958144725292083 -0.92133486788953466995 -0.43673449342702319598 -0.071512640200387347922 -0.7355999721769019839 1.0227557561263904695 0.13660313282906672017];

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