function [y1] = NN49(x1)
%NN49 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:28.
% 
% [y1] = NN49(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.9994540660952;2;2;1.99938933672705;1.99808158044454;1.99906600827094;2;1.99824454751236;1.99904532095273;2;1.99831102918506;1.998467971134;1.99872632047447];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.30003183745509759417;0.11359000218634850154;-0.4278941971380225695;0.023388459043389545761;-0.11702633200327952623;0.11446730346739705797;-0.043434370097056744509;0.16246001192437378724;-0.057077240208905186125;0.17216637073528875956;-1.0827934883983145742;0.15165855046368478809;-0.1694153933138676793;0.15992037608074888277;-0.28144420293157745849;0.069539744320917362685;-0.28752154088853865144;0.67148135110984485951;-0.29202424485424638423;-0.063083328296428847759];
IW1_1 = [-1.1716958916690272119 -0.28796684518036130829 -0.37266608724856942203 1.9207817031450853396 -0.15082241099224408254 -0.020815861934160673902 0.11207434137365114568 -0.13826914254070823973 -0.070981435518078214719 -0.11461745632599326661 -0.35269547396680783757 -0.057602253878790941888 -0.12595786776420433983;0.11457002863686405192 -0.11718357071354676413 0.019705806905374049359 0.22375063872629188944 0.50764190939132392 0.16821542530813715643 -0.35209718504844605969 0.11928184831484379969 0.0054666597577794030435 0.041163324287887312347 -0.25189698449478403752 -0.067836200006698085807 -0.039204964747477040465;0.45888781245359561156 0.27084030144392445827 -0.07347778275499114442 -0.29855080444644360727 0.0086039983907136123598 -0.11709738364486214912 -0.089320733295861029499 0.059029461052559815959 0.0120677294928984713 -0.077196622143699322738 0.023062812685929391654 -0.022715191703980015803 -0.0026313836408872843739;0.51840846718786015579 -0.25032298105051609127 0.27843569566209802613 0.16042114926457423651 -0.39988589344980451923 0.43997909985933214561 0.25626844316479691077 0.049048787000268795344 0.34838951343305202268 0.5170734334607378635 -0.19814382982016146006 -0.083760514827348184208 0.21036354782227942306;-0.25816613201554383661 -0.45843201343826894822 0.037454939378715378095 -0.42527558651969998849 -0.13559363677483604604 -0.1570278262796129809 -0.51777418334431746239 0.3706865787368338605 -0.303173159368942291 0.44913489412271251355 -0.30338311027759323757 0.20795938058399543458 0.13991808918471415613;-1.070691970344557431 0.1543260720909350181 0.42602059910925826891 0.34216094952967318621 -0.14217706713966904442 0.35315905379943141051 0.67819926517012563139 0.20391997533794464781 0.30877684297007035319 -0.2397129669332261348 0.2606763483001215409 -0.0015838398327542549567 -0.11028536648164877221;-0.09386567759947053724 -0.056189009460781431937 0.12153950727392158482 -0.080810668322601558744 0.06446979817716198824 -0.41954332166916769031 -0.01977463096560876793 0.56153050873872945026 0.11803180373126941072 -0.31766954734098540314 -0.12369551141679191453 -0.13742085731522099556 0.060855951004375762048;-0.024599457957538310715 -0.31230239481676158597 0.29352348779217035668 -0.14129221501087257939 -0.13811796609483228959 0.093179834233718519809 -0.33366103423119453097 -1.6485320489866586158 -0.0066407968393495265397 0.42631234822945729412 0.12854882491954741641 -0.015423985094500120702 0.0014604114659256058235;-0.31486112544272992775 -0.29774297873763616584 0.10361039519594839042 -0.069050240699883094764 0.4408906106414518522 0.47355813864282397807 -0.12749943163505589316 0.42566782720220175396 -0.11348495882855981487 0.20304458803784480869 0.082061066202039331663 0.089233056321681852863 0.044685573715528302541;-0.23278366416723053622 0.70069155179448006088 -0.33731773477649157345 0.094171169687817371519 -0.49302657801104382562 0.13894306150255492094 -0.098510307446282224797 -0.75863664687015786026 -0.13921285969893695822 0.70388382982040975655 0.25711038399649027708 -0.0015820907782052987846 -0.039246567214329723672;-1.1283303406535711844 -0.45044396763870997447 0.84207835883035853097 -0.18918131045816563574 -0.128620196019375993 0.30728864823094093062 -0.50752867410713820018 0.66622533451715393316 -0.23412840134199575104 0.51669958404832549981 -0.36259679074636341767 -0.21963867886427138054 -0.12436780631037677725;1.3462975978673823452 -0.10802073073414225624 -0.53363806398245883233 -1.0880791176513016083 -0.00045975550039489311571 0.11436920925711892283 -0.1434111591159441812 -0.5485953935333925946 -0.082960710590664718533 0.029950487250645974058 -0.10316680123891304455 -0.19445537750972802238 -0.18922985551764759382;-1.1729664641912387335 -0.46424768824344053719 1.4768421950056214165 -0.73031376798771452385 0.22384279506522375414 0.11895073731109051529 0.087970578067167154446 -0.77926800179268229662 0.10678689592855952761 -0.40854436574552377426 -0.32047220509811991107 -0.085723778181372278784 -0.0051218128304567466652;0.44176625120849272177 -0.00071448775355133307269 -0.74894099913372269572 -0.44910344567558646567 -0.29691216286799049229 -0.52833666915922261609 0.12863127323320364126 -0.32351561916920135165 -0.054267554398836910512 0.067873286997628082262 -0.48606344765860215462 -0.31257684555582937458 0.21435978796981761119;0.19231112263642924365 0.85772362917804290117 -0.12762270262065292248 0.030821806941680039704 -0.86223038595764478487 -0.80897834896551767958 1.0073043029820900607 0.90964439842703992412 -0.14405808546605450293 -0.54983446576323669053 -0.11696328075585185613 -0.13954606822039597458 -0.15056848908820333333;0.028924815661054925631 0.80637487521801831658 0.77559991823141427592 0.04746371235235942454 -0.29213269553477860807 -0.53862208946254708231 -0.14029400325895446944 0.86041372790431736028 -0.14131586483620472983 0.10343353710961514236 0.057406954180057430392 -0.021817731356833756917 -0.015254111780341894286;-0.54572692428753377669 0.38494155425468606913 0.46251689910379290005 -0.4386673849527960023 -0.079588854369654099541 -0.083492649509601890578 0.22769655570207561479 0.21155415672461003185 0.068619635542193246902 0.11976239564961203843 -0.044154742396187766917 -0.10541498842996846474 -0.012352633254254272166;0.055539224322615904361 1.1303705031963100502 -0.10229681095347108177 -0.37524396567691181703 -1.0838150687663610228 -0.81469645306707816346 1.440137213203195099 -0.36680707743866453052 -0.19644562917022703741 -0.78390347628397993862 0.029287745363900320045 -0.11476095922034210139 -0.19709871717877405772;0.26005449680823344849 -0.86034683249216947765 0.0094234145882876804712 -0.42752522160626649228 0.48526922166031472017 0.31222885043496001467 -0.26438242689853708178 0.50741412016072839908 -0.098595571368848916594 0.21303542266235131475 -0.036007664756162259534 -0.050337649211208868039 -0.028988495586364283058;-0.065284208273756513807 -0.31173186759026072501 0.013478954670450953024 -0.17196658562394387393 0.58608624431901479301 0.22225163180384047146 -0.69527261782967542114 0.040811442760934492358 -0.054966409273827183946 0.11590304749458352207 -0.15013040377260700176 -0.025763331521861491946 -0.039652383421052192491];

% Layer 2
b2 = [-0.55014576835847062153;0.0065772760004757556754;-0.18963389594335294808;0.056987732523619903657;-1.2885177637499181102;0.065108079305973606821;-0.3374904379891837114;-0.030735249100467095656];
LW2_1 = [0.19887159329271791863 -0.23741652602640606062 0.55920286932988028106 0.081509749695863220609 -0.0093805641651079942067 0.073591310596362188856 -0.10992174264544761486 0.18144413565932412946 -0.24023794444312393526 0.3837330492635162571 0.064665798547444613975 -0.48741827893472661204 -0.087556977504497507381 -0.13150433462963356801 0.51341273939516851588 0.1531979424049862426 0.0064582514607206230328 -0.20808474223805004111 0.30235951958167128772 0.29730078814278609878;0.098550572360880300815 -0.45060902958994114575 -0.017812745664503967097 0.23577004209254501244 0.20150105748291455288 0.36314800881733733107 0.049435823180133689481 0.14199956933311469465 0.06661618015640879964 0.42252155333919244606 -0.021535363302843526373 -0.20845864165604857665 0.32222915868723311927 -0.0086413702811807147769 0.43307219386520257043 -0.013511019053455166489 0.043239136109256101625 -0.15378130163287881405 0.39594286012400276009 0.26146487316099736864;-0.052541060546635140471 -0.54849803879313141497 -0.69749268565857791558 0.31247447189268517942 -0.18447558332772392986 -0.069596427832007298764 -0.256195670129478692 -0.037127804498422296853 -0.047088958723233410864 -0.16002785094613317107 -0.054321804099612465422 -0.22486403196521143055 0.053285582274833168115 -0.026050772273799469025 0.55294486173915713412 -0.24481697542664829226 -0.04494151901236325547 0.36085415147704158478 -0.21628349505602989522 0.49291079100728041329;0.05661109540990041239 -0.25219387202171072992 -0.0080992866340464826636 -0.35719539283305884458 -0.15101906766395817305 -0.17036719947688552801 -0.045897815467681166157 -0.21008368323007226008 -0.0096007030534070759986 -0.07700052731280028484 0.11367441491927009201 0.086552272835408686213 0.028191743881949400674 0.084984754966048667302 0.033516426498160924174 -0.10816504617900128682 0.24453038952079600432 -0.38218979950851844629 -0.30439526520646059682 0.13564935974801925478;-1.6187173160962304852 -0.50165615428715371049 0.32531629184587151915 -0.5878484738839481194 0.44006968025400844402 0.78828630587397785501 -0.63639661131826130713 1.617941944411798838 0.2397785976549110798 -1.1434578926105845031 1.5317523815510309948 -1.0771768814095741273 1.9948154919936658125 -0.37144664273726951675 1.218459731023601389 0.88917877214675578035 1.3818465120393388368 -1.5239637194885664595 0.51193279328187313304 -0.5538380578809337651;-0.10120341259968823611 -0.018228979353193033169 -0.083364261141565781932 0.031066068474718960823 0.22950470510451140282 0.13471460748987418299 0.3410480137665979905 0.015100542427055600736 -0.079203938353398367633 -0.24634260923100984031 0.5189085689048071881 0.021491121781721460315 0.19936420448222494373 -0.29666347474100140946 0.58814544734028439965 0.0080886460747007608851 0.11339472250639681672 0.030628009190745986157 0.085966784128141265908 -0.45212781169237042933;0.014174527041805842775 -0.016215479583011636888 0.17653264662101858518 0.049027793746480759218 -0.18432025670774140891 -0.11069365647307174982 -0.49446294092766074346 -0.060993914461832374196 -0.095029827743156644604 -0.067288266394196466491 -0.10424963779944199094 -0.066865987842378446504 0.096727329917283430549 0.1334446989214832513 0.38999933798482844249 -0.2680147420607528197 -0.18395045778478649745 -0.14210153205514455821 0.043118855846112420782 -0.48466343052039567718;-0.051855619384669530969 0.070601749050064166457 -0.34711454200840036188 -0.26738744315881018965 0.040624109222627657889 -0.03917522804364805683 0.19889062363910400832 -0.052542691162546270589 -0.16119839028026131111 -0.09486557650528416874 0.087104647689049630177 0.17604431947161491845 -0.082747603507901459907 0.018188233623762314201 -0.29951853018180052413 0.13096629006071500156 0.12727898579359991915 -0.17668900279904617423 -0.0018910956468912755446 0.020886766782251393187];

% Layer 3
b3 = [-0.59485070909855386478;-0.39183652799876422801];
LW3_2 = [0.29446302958710302011 -0.53058611476231243298 -0.96795335370302704181 -0.47564462410609242848 -1.259970722963074552 0.3364915160768696123 0.71632492543150150244 -0.75342729854136170076;0.34608727626983842862 -0.40554545585856999201 -0.69458966077655948101 -0.64395018849779583903 -1.2211451957708505667 0.29757977820061626284 0.94352193277643858771 -0.048507453888904705774];

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