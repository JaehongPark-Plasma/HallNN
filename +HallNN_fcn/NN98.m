function [y1] = NN98(x1)
%NN98 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:30.
% 
% [y1] = NN98(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;1.99801779610965;2;2;1.99893506117587;1.99821187621912;1.99832469003557;1.99805141435741;1.99930968636064;1.9988765081976;1.99820464068164;2;1.99887302773195];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.13232910086343366296;0.85977173515554694561;0.84445049377038339244;0.24415213006098529513;0.32057279179872621988;-0.44849413720485514778;-0.22331583885762548713;0.17667730357434968669;0.039291533542586985772;-0.37999835072056870056;0.081613236251136492272;-0.30665294654131081131;0.16162045194358612421;-0.072974768301515891;0.82821983603392013151;0.18269322888923905657;0.14753787610980800826;-0.06126385479476163104;0.91116889430667202809;-0.29683530093992127608];
IW1_1 = [0.32514668248513733939 0.37195024737751869459 -0.28160354292733474901 0.25558479938713646318 -0.16624439727762380103 -0.25197192637060306097 0.23402614899422180539 -0.18530130545169795631 -0.053660612295890343371 -0.19345297016705156889 0.32673654239660232523 0.015623180556021543444 -0.0060904181605133354827;0.98994714425752805553 0.50933750543979716685 -0.58738722913856522556 0.40571034123061378018 0.26794406083229671545 -0.23545323666965914389 0.46740796351796071173 -0.91632632287974125074 0.25046993412336437945 -0.63492546295186302885 0.84758017580412792213 0.17971891929203814842 0.10345872376636684742;0.3263293433893781903 -1.0076224697432816235 -0.68148530037458432407 0.14971554866528838379 0.68177046528769058575 -0.072258704953823194028 0.20356763231402891234 0.90353198210719809502 0.063157789477639975084 -0.20027067504711640322 -0.065356328343062125574 -0.0018421699895289438652 -0.095348957897063255085;0.0022291627770607384817 0.15775049906898691576 0.5978912933886215697 -0.84515954747136634584 0.33666985274261940209 0.026923342650103036788 -0.43567764393075142371 0.20001306307773611404 0.01731142024752518882 0.032366080861331671115 0.039077780645760569478 -0.0097746426790874824803 -0.018498282211382839624;-0.329740610649819188 -0.7904878946170259546 -0.17603024246434270661 0.010261207689869772039 0.63017261033926563751 0.50630206230769803089 -0.68471828635682074893 -1.0622291963061580855 0.15849962205755516176 0.52267150281963026259 -0.097292093947113258978 -0.04858294193359859181 -0.02010334541275266701;-0.20413403223303894229 0.14706409650983132131 0.37302535181062485004 0.16248944414125246483 -0.53983083992349045754 0.052045203533215581015 0.23804843051033994228 0.073816495558479577888 -0.31112264056601807161 0.53989027016405044535 0.0021649235985406285396 0.10249267619271429375 -0.027553756323077237267;0.082516525016082115096 0.033211994960753665751 -0.36745687648187530527 0.032696393015303130969 -0.63721460258239792651 -0.19899476633714599516 0.70756943026218122039 -0.1033157448359760533 -0.056379990015566851247 -0.15721931715661904838 -0.15953769767687031589 -0.027678488703168410007 -0.025328593921960400326;-0.43890863928255197113 0.57520754514112892863 0.872585963792477326 -0.5858400345340997184 -0.39305790376960902943 -0.62834664492063585506 -0.19292538983877496883 0.13276776864292319513 -0.03942321193508848548 0.014513246316947581507 0.23339195401726528512 0.00064498479577097811433 0.17423592430216175431;0.85515919198654644084 0.51801617108422859381 -0.083347519752578411167 -1.2621582036151424155 -0.15661098353758590163 0.37364937857949220845 0.10622293853017909793 -0.10162792226679515095 0.39074173261402850299 0.0798485133908450051 0.1053468178076283901 0.081792190378797882677 0.19330271939297813399;-2.2827285788283435686 -0.16291867412551924721 1.2099065467870548929 1.2732158217425593172 0.043734193882920435748 0.042288674268535934864 -0.086374760664853708625 0.27446593936480367937 0.01513648104385182519 0.069470035691006876144 0.22013762082504961803 0.057468966041434969838 -0.096487066184835990068;-0.02943005694694698221 -0.20662771155556891589 -0.096447181647161678297 -0.39585486752183246528 -0.1875814171999582447 -0.62318359970474579423 0.51183713003731634572 0.33629686197129088354 -0.52622344442717139401 -0.69016349254399089741 0.34868275709659496409 0.27412008982280017255 -0.34661728764035676376;0.055926120882120629318 0.28015703491687554694 -0.17298647622158697112 -0.2200848609528339439 0.083588370740145884197 -0.095442775131826779189 -0.06674880795941470768 0.005401994986870966263 0.010913497214400373625 -0.19781069299226547442 -0.26860390721673299552 -0.024003789458476858243 -0.0055443254416309194557;-0.27194086566416714179 -0.70745274037645400345 0.31728834639780523208 -0.20737407061515467088 0.40559302146239833986 0.56819287902748427666 -0.29411856966280730541 -1.8823097660728584923 0.069246582745119911184 -0.097835171470178575182 -0.40085011351774235688 0.0260947334336445666 0.15835479921728209951;0.1968934575532735598 -0.30345549986513048468 -0.32013290363898655455 0.71007024824574382915 -0.037852294005788614251 0.1268478679299411549 -0.56419166336182458021 0.20338846679404071205 -0.071816536802338126755 -0.59397337802620719938 -0.21180549467340575864 0.041346395561848298117 0.22382698291443858807;0.12226917495730789398 0.071337189323168639921 -0.52552582170735995248 0.54192866008494500196 0.039755323558721078481 0.10846590465071617726 0.057719166548490435931 0.0051895187499258876168 0.0057800770749846059823 0.074143706362807468291 0.010009261745103347277 0.065055992053708572298 0.028459506275772390033;-0.080256204212017609456 0.0052356784826418454465 0.25678857823037692931 -0.074779994563838492194 -0.27643435287459161565 0.28479603050114893348 0.66609225128210436395 0.43257162816495803304 0.15451103303162838642 -0.30638091441866399656 -0.11184606440237288005 -0.11546627048717385267 -0.094161103503670059989;0.36545943049320001039 0.62553209350284377788 -1.3624916890972289529 0.60933189684919464302 -0.50556567709770527941 -0.0015280909864125221537 0.14150363231100068773 -0.032170667774629915781 -0.082692320564110674552 0.6409855477302561777 -0.033622169563986163687 0.11055837870092927178 -0.031404122579718918296;-0.47097875583814857725 -0.20874857903917573565 -0.13175088972935422582 0.69222628988885148704 0.21853401121159871101 0.10236278161425289845 -0.30948952423491143948 0.080747643497279114611 0.0033146560734786719576 0.14277939893247762471 0.11760293905323390695 -0.0016680606540797719706 0.013741868395001050063;-0.089288696596008312878 1.2313204504513821647 0.049996807590096010998 0.13536976623343724424 -1.0225053236673118384 -0.60013460457119260916 1.0319901378703812966 -0.72064730535784893295 -0.07897069763682440513 -0.58953532315298140531 -0.0662741807861897958 0.0093086179583500988644 -0.16602473191699973487;0.020283754270009491522 0.031443239232903187619 -0.18634602418886728192 0.48444591042317602936 0.21886122310816361058 0.057816055083558678951 -0.29984979471204514789 0.19775610738996957094 0.03286481709621408287 0.026892646424588672466 -0.071948909419737305804 -0.019974135529246036386 -0.0016879054016046008812];

% Layer 2
b2 = [-0.29672776527599864504;-0.024520250056873298117;-0.27197839733580359001;0.29346136439528441864;0.28309764518286972823;-0.16418251877459760779;-0.13002612167905919227;-0.95002077647353999623];
LW2_1 = [-0.27857174526502170853 0.1126798259042476047 0.14067930152778848618 0.30478004549511383825 0.30439769746866801592 0.17798574709358724877 -0.19584990865164256246 -0.40686004190801267066 -0.049731309856866690033 -0.32348877063864656112 -0.20282720080785829331 0.11955469983206423734 -0.2095335119133166224 0.14230311798437705884 0.33611273019791804151 -0.44919185769357317106 -0.21917405749098609058 0.69372577491134679306 0.45426442266078115306 -0.61345143232817012091;-0.035924571452803082117 0.038867574374281975425 0.094179969253338033375 -0.15116169093793313527 0.26071566882761787465 0.0012163397957366435531 0.20630485106560275677 0.10666622400136172022 0.059526236641248120307 -0.048752357177068703087 -0.17825239254443792092 0.26808316806733195614 -0.37615701247648164474 -0.12759422937988357249 -0.40361784049604287894 -0.04547936285447232857 0.14302477093542351194 0.38584250337299536016 -0.10042602613139255119 -0.17008247836642059747;-0.16301375054951072752 0.28105984384298482448 0.086921434480903947417 -0.044363079729197646417 -0.81155382157368005647 -0.66611463969359452353 0.42415914073708332888 -0.3353710598563638956 -0.25852016185899628908 -0.23432095777201886189 0.00047741906187406986384 -0.22339652330134415759 0.3814977440454330182 0.10212067116532164146 0.32515549866718435057 0.12690549306824511211 0.5888293490501382843 0.43906127077631135203 0.13761579401633711184 -0.20161008126730212142;-0.22739476932496449635 0.17232716215835866547 0.11195852325883523526 -0.36525719381252130802 0.01894488324165771731 -0.091860543953428089314 -0.18466480899566034557 -0.50024648221239864654 -0.12211346875905461362 -0.32603726473920663365 -0.14846069219956695151 -0.44548313580284598645 -0.14148321865933619068 0.12621593135389314044 -0.34461266324345946055 -0.46630802944130689269 -0.01006855481479120018 -0.32926917246696990604 0.3200469624809298419 0.4196786034885431893;0.02357713371063224761 0.14045867232525244273 0.22460990847680981086 -0.47046684695564733314 0.23423772332059716872 -0.35182261105940748624 0.00033862593755935518902 -0.2503191502620271125 -0.10483427499684042083 -0.29311866076580073148 -0.19794648067212361475 -0.039967789522354274512 -0.48047832519476391111 0.10934028133947153172 -0.059852329558572087143 -0.12596967991611676685 0.11670252605991651706 0.19504426812159067484 0.48125618700158995233 0.051049090086348600237;0.37675470613480877002 0.42072572482111780801 0.1666749662624644468 -0.44169449819210454544 -0.20756130493156865757 0.12406523594837257085 0.033546635427078273783 0.063455314010371924316 0.056404235652293820136 -0.043444801565542742749 -0.12407665191412971251 -0.10960008824867042976 0.055904293166375060298 0.05298554099915569332 0.0086348257300374806067 0.097784847735625249343 0.35719525202916085904 -0.15907153506908353058 -0.0072443758589024551814 -0.0094011173740042031088;-0.12305269248697406348 0.31786356983942448684 0.049797488705543589693 -0.14230529951036849368 0.046457535481775509567 -0.069186375911130998384 0.42414969212750258665 0.099000893123527278217 0.065407491385886931679 0.18569059724255615462 0.036609506369786798397 -0.13945420795363827615 -0.13331137935467185507 0.12459038539181463423 -0.41108745177928679437 0.030218201985798331943 -0.07895324840176042025 0.17258278002339885737 0.27709470075393477506 -0.36328522112853656179;-0.22163417654801614853 -1.4407743228692555526 -1.2583052376732570821 0.6763007514599082004 -1.2977879041672770732 0.35848223285747587985 -0.12535048797912587415 1.0754256673785553833 1.2152571780278369484 1.8596931669410206034 0.61983424343168591619 0.18119062820613940357 1.8349591770325386353 -1.255179082971302984 -0.32053004983369515202 0.25457507518015320969 -1.8427700132787916409 0.00117028079195569297 -1.864884380935351027 -0.15329763894876288766];

% Layer 3
b3 = [-1.1402309775288173022;-0.78964379286096508803];
LW3_2 = [-1.0333953128493220674 -0.59864834122957411022 -0.65304939073507939362 0.90723050677745342707 0.53208179439916281073 0.47125504124174366627 -0.54880336280348551714 -1.0661159612856268009;-1.1208113841565163771 -0.27902657525688323581 -0.55199314711440905512 1.2028083073436588446 0.1409219873573527293 0.2587852729081799863 -0.24641885010456099359 -1.1271871009340181935];

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