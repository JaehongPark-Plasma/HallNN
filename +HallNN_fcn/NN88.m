function [y1] = NN88(x1)
%NN88 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:30.
% 
% [y1] = NN88(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99865973416029;1.99869075396185;1.99990325001269;2;1.99855662656271;1.99889277497243;1.99944116623018;1.99856686410983;1.99944887444175;1.9982101880239;1.99830878802397;1.9999313611824;1.99942940410049];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.32610040063389089671;0.23925582654962487816;1.1864092804544248416;-0.31165548026540057514;0.33852116270339815962;0.29447680003828874407;0.075271602254128003318;-1.7098818725626043769;-0.47746546965693703601;0.31758142428231367349;-0.41680904387963396696;-0.51674022696067223315;-0.3375896618846124797;0.46231136361504981958;-0.50879529491287411869;0.5635478350779974388;-0.41931459194842440885;0.71395040733111825304;-0.19639784764140366669;0.067116157780073404382];
IW1_1 = [-0.30809442343650206286 -0.038517706410296233266 -0.26966693649563272528 0.47284705002342120039 0.33365350965051387977 0.065991740658973840894 -0.4984219152226759264 0.040197419469410092252 0.0040110847972816814772 0.11029184044081484506 0.042360520924665100162 0.025046441300766591337 0.024306483299116277141;0.33626354699637589496 -0.51695334954942129269 0.25203691024674884646 -0.15154846881397410208 -0.25798839315167665376 0.27661125767854505275 -0.5177718437113664196 -1.2268945067329661835 -0.27625113183419824914 0.9835995308796805503 -0.012833462571270111419 0.11612409783020209708 -0.16249317079207845071;-0.22865825981125234567 -0.97934640082310997489 0.10096724435136919529 -0.032998814598657610908 0.71577170695385616828 0.63624269619535944731 -0.96894581664523826703 -1.6225876240596799338 0.054819162316886528574 0.52686801493594181967 0.025847467256941085434 -0.00039434168545010041783 0.049613031434096538375;-0.1193444887408422056 0.25145715448524624813 -0.144646133662545473 0.37469114351717508482 -0.0065375828304244966438 -0.21003229681256593842 -0.52603180947477401475 -0.15624603754294716373 -0.038264483476691389663 0.30503043617857961145 -0.15164676282138855878 0.017051082936203755536 0.028010846177276952501;-0.32693477947329147026 -0.4066479459665174323 0.75508797404293237499 0.45327064233343439259 0.46885716028367285801 0.19540744123070030724 -0.099326289135863610191 0.18974938185661704981 0.271056775733848343 -0.35088201139245528415 0.2872280191710098185 -0.22198812963095918005 -0.11498476610564353773;-0.31080088839684788526 0.090885146726336379919 -0.075829312194659062585 -0.41006122907589431792 0.35250851004993466464 -0.35273974835849547516 0.45718910850370314547 -0.1362534275500864589 -0.049543798566667969496 0.44097942112671162374 -0.030103609224459024207 0.56697592847575151165 -0.48676004979202586176;-0.38177645510069196932 -0.10305838760608382332 0.46892186482886577537 -0.45689344734710984897 -0.35064704090331205721 -0.050817441237725760605 0.32416936289940256888 -0.0069751365237506789477 -0.052268335322760851869 -0.059820893559398138906 -0.030275392782968153782 -0.030131520988662278959 0.03036341648922104286;-1.369835700774021392 -1.353411190031072131 1.0433531407974681038 0.058322874354119119689 0.70588301161931821248 0.47601293469956046378 -0.83843612386672794479 1.0795338467191026854 -0.092492951063736852935 0.4895922635459866723 -0.049503645089181105843 0.024814678759276589254 0.0031685695664151057767;-1.7796431324851806899 -0.25521175461098327686 1.1267692132485609413 0.17785442060020584254 -0.20980951664138583479 0.24860127135808529619 -0.18289822224344096102 0.33404793798198811894 -0.098084187161193506221 0.61904432131463149602 -0.52243409801277174154 -0.23942140031945294276 -0.082267647442406832692;-0.38298887756148947137 -0.70560960087037405408 0.8122108717986444848 -0.21989749645637360342 0.0047771229737936619919 0.06683113960386249941 0.49315483126630255128 -0.26892466301564377229 0.19699520349714133571 -0.69281896113115826186 -0.50316715039669968057 -0.035358648188567945447 0.06782630490406067747;0.22455568144382692664 0.70725314000198036268 -0.1121771626463182181 -0.062439382692283634801 -0.52217505820716503262 0.13331981941375481426 -0.068984884014827704823 -1.4561933534135835799 -0.075215389804869581258 0.15319761529324998905 0.39021445593247383599 0.052247863590391782629 0.11245953359080232414;-0.12353472485914114565 -0.64460890134180348632 -0.68061720863428198935 1.2270077521065332338 0.29364863885925485132 -0.080102995625899592325 -0.55674562201597421751 0.11265583739289702914 -0.38034791985695642902 0.42856341611094039834 -0.12013125163146309848 -0.072899745892081285747 -0.23462972461596168228;0.62709114918415131967 -0.90228984543719148448 -0.92524789805351181915 -0.70765356437863524697 0.16353257561944203879 0.4498054966289104506 -0.017698279964256481006 -0.52640353508638060287 0.13972799755037604297 0.048961881486213630621 -0.51341018381319236674 -0.27192725018362573497 -0.16287190240655768925;-0.25392712148785562265 1.0439752395032719789 -0.17955109933946505896 0.39957182453810791145 -1.3011619403045462384 -0.80848090218954360431 1.6878523903283086938 -0.076531938993338516353 -0.16168803019137037436 -0.95106555311731344471 -0.041749870548839669582 -0.13989825988112530863 -0.27316190210306179331;-0.58965035490222283965 0.044100677460848676514 0.41540665738646065375 -0.25163434508895343722 0.23754219356980427191 0.1987565666386326424 -0.21093424870460511511 -0.65201871161881208572 0.090181301499193719517 -0.33458995489551873748 -0.17754779922646860002 0.03311111847313619877 0.076613617365194522368;-0.25902343137848360399 -0.17065744364893964402 -0.81994967312745103971 1.6737362026804964987 -0.34351110558191361166 -0.034738152465149434167 0.56433887666215887258 -0.15035224096594182464 0.17159395584429393278 -0.24732952483893833051 -0.74841374499129165176 0.16761782791690923999 -0.0037066259502222245401;-0.3698515555458806725 -0.41267300485886548866 0.12819425628963901365 -0.28934008028504187848 0.70291775922340926641 -0.28200115873843850656 -0.04454101428419064701 0.24010262822951947137 -0.03472151686340652077 -0.69733951809342131778 0.15773015645069682344 -0.020245626894394970485 -0.14576715505761828062;0.42662663913696519113 0.5854924654401858497 -0.34872670109501913149 -0.081807388066455030695 -0.28402064035509433371 -0.36645925213066460335 0.41951287607339232677 -1.1537501880954774691 0.09248406345856363675 -0.40730252420913537481 0.16860631106059242867 -0.030876964352267475405 -0.1336135307896481883;-0.48954927467324177437 0.19845431632688073642 0.045855206992337756156 0.055363158908031645522 -0.11563730480420758573 0.0011083053098470468587 0.077036379261658544104 0.060800657150149574337 0.038475055438755810522 0.24771533933493875002 -0.15838275358004536009 0.021680622936992813288 -0.039698469644026929126;-0.51363231286089960914 0.11736861169239759528 -0.099850991036725511574 -0.69258340254896388899 -0.63623151522271848002 -1.0079441463709306603 -0.48978156095457103802 0.33820564809531333816 -0.49495193091961997567 -0.10408119139193972025 -0.39867612099239507772 0.01099778607110945064 -0.070572769676649632631];

% Layer 2
b2 = [-0.2333913132844035454;-0.12053072066999345746;0.20125320277832842164;0.63940479819332518563;-0.15071435737317123715;-0.16226570132939172053;0.63853387081419554772;0.020035543457053876792];
LW2_1 = [-0.27555945909845086517 0.36903876858213319201 -0.076772376181106660731 -0.22411137218217633271 -0.09857673588024012945 0.046139503001188478248 0.23080957156482764803 -0.12648519382558323643 0.15601379965040096387 0.14271056837922038718 -0.21051046748127760777 -0.25496125680858500173 -0.38912598686322419361 0.27413460244744858318 -0.037092800560737076987 -0.0095264734318659220758 -0.0074781081095747128223 -0.054694184692313248253 -0.18591280480380334428 -0.12889819358390081083;0.13243723679299646001 -0.026448078393547257148 0.13138283273765943848 -0.1521601392930853891 -0.03269735406699992436 -0.082992908466472017914 -0.41630899723838415971 -0.22724333857504705469 -0.14045678519021867481 0.0017601078333010656413 -0.055103931058257178688 0.086136112437769474237 0.054196166933002810906 0.31955200889424772992 0.20110812090495935878 0.043377724888699752126 -0.10323740329347318168 -0.38961913823741778362 0.025808654135016650955 -0.033626926256775400981;-0.46823660524519583603 -0.056570439710091242891 0.037942634794129279086 -0.012919915868724764565 0.041811212634419205048 0.054543426863695024243 -0.58123548494960886046 -0.18694087044095536454 0.43254975499598874444 -0.29588370724184959171 0.18166958511529998632 0.018597242257805457477 -0.018240453446238435314 -0.76098265242525353624 -0.12444563681324574689 0.061543505848706506201 0.343654963611777442 -0.028349042953744268969 -0.26765801116767412582 0.13500685658847919046;0.26727467516828712357 -1.0097751710670754832 1.8135285602565869301 0.07176295879603346306 -0.72494499767063724782 -0.50096847933511001738 0.093295924479752948 -1.2995714990224782603 -1.6529853772419316194 -1.7226506143726321163 -1.1202738728459351591 1.2958124759212261701 0.9251595282782102414 0.84953501092686978247 -0.79386942746138500215 1.5393792640547721007 -0.5939251844841436645 0.151091930499967525 -0.39409754488718296006 -0.4385354512073914024;0.12277190983280519831 0.30551636499995837859 -0.12920302351588788814 -0.0072996429482067460431 0.24431737200627093309 0.23196945048728206951 0.33049723966223626404 0.10748444460208973761 0.2723657061181594119 0.052158904038019440752 -0.18074791125604328101 -0.59440381692629273847 -0.12761328027603607849 0.035341460152695867991 0.14967063606949049248 -0.39367515224803878437 0.002626747334428991467 -0.23071890732233898968 -0.097441349492616111116 0.13341733713100401015;-0.28154600820405650818 -0.022519363241530705733 -0.42864606252593645541 0.27031679880945930217 -0.19652927919506396615 0.031381018262557167087 -0.38534649671459225706 -0.038737915608760588315 -0.076436469882698074296 0.086855565117456373647 0.20803979053629484408 -0.027521904378691029736 -0.023587851782147711138 0.13912452147879872388 0.27483660613102933734 -0.16139207569445343515 -0.15962093459015613828 0.51207699719088917156 0.011967008416134542956 -0.11054304674067788605;0.62528554469104713487 -0.29333216877781609355 -0.10744753780828909806 -0.39514203068990555456 -0.3120634081457328568 -0.14807538155464133878 0.055658696996633567899 0.57836345021175050718 0.26314705762103351994 -0.19814535149934420222 0.35962664403689575954 0.40018455396383906253 -0.3461706672012439534 0.079141389992758243155 0.062022856829736398854 0.56910530957029936694 0.010230134819895510961 0.47938968355721139902 0.36981155684539862571 -0.17014025524971232439;-0.2731348651644738279 -0.061450909358961025575 -0.1198634465101270502 0.17646764276548440598 0.22665920504925043955 0.0733031192729503428 -0.29602689657731351414 0.4783587513112877998 0.11527004874801624612 0.018685087638236490964 0.12573223817001785707 -0.13350558390766689709 -0.060320956696623025117 0.22205826699790540135 -0.11805204567274359129 0.0088230846711180727082 -0.17340712702892321917 -0.23535691062207331203 -0.16586440088920084701 0.10859086771477265143];

% Layer 3
b3 = [-0.4091535413703286661;-0.40827984581303250478];
LW3_2 = [0.65345263625342264735 0.34959458936906245841 0.63187345862604638036 1.101051938799337826 -0.70405428649315326961 0.43255473074521083587 -0.41485347209658784973 0.73324871326150342643;0.61012361609942777907 -0.051354067794573611538 0.45496258063926159343 1.1201410743500330724 -0.79583670190136346356 0.24265361152366773934 -0.44947859203069084888 0.78818364627390324895];

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
