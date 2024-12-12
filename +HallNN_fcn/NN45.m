function [y1] = NN45(x1)
%NN45 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:28.
% 
% [y1] = NN45(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;1.99811757202069;1.99905424654478;2;1.99833623824399;1.9991963058358;2;1.9980238739958;1.99804617217612;1.99977600411302;1.99822893473187;1.99877571969398;1.99850699997907];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.24529062490127606644;0.06740520900246781133;-1.2072872656133777358;-0.31587489718024508445;0.35190807812976748759;0.11760411822649297076;-0.062394369356006347538;-0.083417656474462476002;-0.43722146174753823145;-0.2790098864904779763;1.2673545436710265122;-1.1382934351882956214;-0.31924578873016834502;-0.18251608639503252851;0.43268556075629760471;0.074072542619695438959;-0.90402004221217724389;0.077230427743679205821;-0.08648397459640635998;0.31136639248046910078];
IW1_1 = [-0.10150859807561037063 0.22964532144736832242 -0.055866581308124670102 -0.21613536175723438437 0.36809041598943070817 0.51453392729492086755 -0.28053036440045353572 -0.067216061846703034632 0.066379005523000148004 0.21910234552308427425 -0.22623441432980506693 -0.0019065997334338164251 0.089312064103986940622;-0.27741613878492132894 0.21472381344135257897 -0.04125669866357865867 -0.27464909852598617945 0.32206328824930946508 -0.41588959159431171386 0.1004641712140715154 0.038726623874727215957 -0.056789816420524420748 0.11318102416424395684 -0.082786631919267661406 0.28551804118301943314 -0.38466225690012734173;0.4234107851708383552 0.90242557718889782059 -0.22072445027298462361 -0.2112590279287136219 -0.71487230980149873982 -0.58626771201637817299 1.1220707394693827386 1.4145447559639550139 -0.010263819449951345694 -0.67177342005290152649 -0.052561360494580412761 -0.0044301754864444209403 -0.0025941894066975820787;-0.64102428490120599847 -0.5032194477130423893 0.27743613304213587956 -0.4159461234823414455 0.43247072820034665463 0.016443770960309030127 0.053806414713128386529 -1.0897243238029501455 0.10406518023593959443 -0.90244539690340463967 -0.22791637859853347559 -0.057226684708870011187 0.089626381340103977724;0.14746795347306906887 -0.057434399946575592777 -0.15421996092180673132 0.19793537231245664221 -0.16908984079494324138 -0.17729290174045381212 0.329956783858942726 0.6557479997045243536 0.061891856658959333581 -0.24739772018957464916 0.22435689008205533801 -0.085413475192372440015 0.040454543467712343607;-0.60220507711269555262 -0.274652730337300921 0.23403318074880069544 0.066867335375155018129 -0.021253663114951693025 0.055267679423289042817 0.12577075719778615936 0.11648761596957590836 -0.019960930718311585552 0.064264891766984452715 0.090653177572945803364 0.010548776683623629219 0.0088478550470397661354;0.15020067881430626544 -0.34970767695204679759 0.16313346796545735717 -0.10929825307377259713 0.1988565361772464124 -0.11388621877322266385 -0.073780503996467489136 0.06857327651412561953 -0.042077453526276023532 -0.3212089401682770573 -0.020128358474945057949 -0.070078750778177753555 -0.013716073277835546737;0.11576597315290694523 0.82433184296738437613 -0.28115861758393057013 0.33642908253239744987 -0.55508664511759653326 0.1419384319386297888 0.15664003080313684824 -0.33277811371834786769 -0.015224999784567276942 0.83039576832453310118 0.09303016339607173002 0.075854280725917724859 0.014836748178492679268;-0.59037488482552447078 0.34020025406384241107 -0.18571919350975024665 0.43211871188969358526 -0.69489607077851933692 -0.43286710721378646172 1.1408500424697478604 0.91583809215139255766 0.15924420393159996467 -0.58635546165286311915 -0.19798270210496335397 -0.059710735617042626222 -0.0072225091726361035499;0.18212857147813896197 0.0099653277973900562803 0.3244201514664042385 -0.14663807095509048817 0.12010032276076161584 0.36711433110815072656 0.18581650158004903162 -0.060925311918857791293 0.014309954690747297631 0.056233501713088711316 -0.19399349622046618835 0.026227261840828852701 -0.013981014950754050083;1.1536643634231840139 0.69414037755499558013 -0.70662302743701232988 0.57698221181094899723 -0.14566552761955989248 -0.18148699048092764574 0.56421991379872882266 -1.2000946191255386086 0.26176425478967429994 -0.53506577204579797158 0.66936717277547019833 0.22591632299509695203 -0.061936360760600772668;-0.02734445770970422851 -1.1062889692698334443 0.3440462355244062187 -0.26656838620481176916 0.99403261800142839189 0.72461624290077408794 -1.3640202734680766383 0.85394401827731447607 0.13196358122234994048 0.84268293514608538786 0.10906706204813559902 -0.030801002039658745973 0.12853042741243400626;0.35480069152492038631 -0.64603949405199267986 0.064513427678226806239 0.037397932383255345912 -0.14253327522552450279 -0.26983432223564723929 -0.01747506484916337352 0.0041183111965533815771 -0.064325361637460012898 0.23283766969840635763 0.20378733732422624203 -0.016682922841099401268 -0.020094278248926841085;0.39020044812053983341 -0.77411355493144529127 -1.160622057870006163 0.46355589445286493611 0.15207481625604610254 0.70247016825253094652 0.12071447078845845557 -0.6805435370546643048 0.089939710324434846145 0.065101113455339990055 -0.15805937340402662072 -0.090548248458394153815 0.016656715668600052815;-0.061809044138778676536 0.42879109383895508412 1.313200032357456859 -1.6149153390963100208 -0.59310167750250786245 -0.1232618076879465896 0.7809585210317926407 0.011824996968611745443 0.094136590126868660566 -0.3030900984410021537 0.28018856943003878079 -0.077926547824879066617 -0.011138178745667293851;1.8012141923914128938 0.11363838007951772291 -0.43474689326209059459 -1.8319308355985592929 0.043746568174720029676 -0.17598794860200414614 0.08799515814908884237 -0.090496411206137911121 -0.035854750539703474088 -0.16969322179611270873 -0.099253078337341815729 -0.045273106163662965584 0.082529614505824744342;-0.4267572536908684766 1.4708038599915396194 0.31624079583814668215 0.008807760910161145651 -0.61445502486569580913 -0.13585740400092760627 -0.018157305775395073122 -1.1451757142406993051 -0.086946271707171701726 0.38384514519204770977 0.044372555890999539063 0.15844648704029559716 -0.025578535814642856111;0.29798843608727382248 0.44625679802656875728 -0.63073850556864852201 0.35500007667403149503 -0.0185264559818525687 -0.41183491290010304509 0.15233186070672691259 1.3274980049185034492 -0.081590138996732217147 -0.28885295262467824973 0.13233971837398236793 -0.030378345199299966461 -0.1780855315350245438;-0.080339551334132092153 0.99452855011993190892 0.16241768078069065973 -0.23962525377896326528 0.31119003445516801865 0.0021279379525385401159 0.12335369566325354862 -0.52400626118064153047 0.1764070664943463429 -0.33668707197820357813 0.21598033208357031443 0.10322836895573204996 -0.038502174680412934449;0.50354593303414196814 0.15516567403052231033 0.04477912066777580552 0.063799289535973455201 -0.36616497795200930732 0.051613258312404756978 0.57018717585991163244 -0.2177280571897163286 0.039861737057417252761 0.16468279463407164975 0.22646301080712583076 0.074638538976810955838 -0.016005281266305525917];

% Layer 2
b2 = [-0.253142204560688866;0.10483225836395564101;-0.75115043891848942703;0.37282466521760154743;0.18991264423065767897;0.59075104588392468585;-0.48401197483545321809;0.42792667350510793511];
LW2_1 = [-0.10643373082556584208 -0.1245418930503752325 -0.0035192754921029539665 -0.17099138300007102331 0.076313412796130908333 -0.37428912432474026328 -0.13263641436646012028 -0.075456778626072251726 0.50222708960159601066 0.29459989618631321484 0.26507319583387206618 -0.16236264539985584276 0.18633441579811377276 0.2168083847045935153 -0.23577002380411032911 -0.09783014702257200601 0.17690070540784519348 -0.030955879668683756317 -0.082452148634849398157 -0.064769360594628053174;-0.042306032712018486197 0.027964867505114052171 0.21899609899099944932 -0.027332738949004630608 0.27600044905203269252 0.18870913597927310712 0.013943480530518611096 0.26848161405654585865 -0.46121651579982569924 -0.24795451453854658319 -0.37972945862293544517 0.28714315575056947871 0.011927488352730594712 0.0067347675924335128478 -0.44464842429099837862 0.19994591095744590481 -0.18240428982325140872 0.19338407971714952072 0.28316705515831175344 0.077200510782491987838;0.60007921233879912837 0.68767162992541541744 1.8960735222542779432 1.1627867399668854187 0.011494181133583428189 0.36548323042500763247 0.16464713057037344557 -0.83919396973949789142 -0.73261754869748918484 -0.045780689332973911743 -1.1473360980356275274 1.856188031181581044 0.037064143772485597106 -1.090499370058930495 1.9782822924407637011 -1.6776833626248530695 1.3299073137437760206 -1.2954784068546805198 -0.78753297340657857362 0.26487013730659053623;0.09376703849414132208 -0.07757176298263933345 0.2496843038800195258 0.070102286748375000625 -0.031138507296007894576 0.29539046020446246832 0.055766121688233652454 0.025524761822110920984 -0.076882066860322353929 -0.42331403540891765491 -0.45916433262328493203 0.46913531251223483665 -0.46403087029431688837 -0.045305005925242142206 -0.11796114383128458725 -0.48926357496494782273 -0.088426556596060509885 0.12121141664668241833 0.36081551155761881766 0.021646077644706225585;-0.01564067120552474488 -0.15643937186241754778 -0.19153366329793528045 -0.24484237541274625483 0.14770107743933155731 -0.69084543738172232707 0.25719848342609130665 0.51041495476594800351 0.23615305921254298172 0.0044631002177173868806 0.59754164282444877099 -0.77225290990223194321 -0.28588486404863000523 0.22282778166253444652 -0.31780297806049628839 0.63091051270549403096 -0.082325916283218086855 0.37816897612608463275 -0.20771983432210824883 0.24844146375719011743;0.28552062948538053933 -0.12826786844706092316 0.15115742583493929629 -0.24283075885195071208 -0.092964580718846279161 0.57460560351063016959 -0.33968442780087215738 0.058601653401544620547 -0.16828846251779736032 -0.15731870800402886212 0.59978453121194086339 -0.5044929166583973501 0.12988170883957270152 0.33886313009524587914 -0.48151364348516972891 0.34963772654617158997 -0.14095653858167761507 0.2844025807567140407 0.053963875656781963575 -0.24537391875050509893;0.046213609895788612669 -0.095051330982092341326 -0.14740492269523508551 -0.1911658904493578337 -0.24043545347947653523 0.42066179840696282799 -0.33378555361620343422 0.335103851525201335 0.28321611537015706217 -0.027883596871633730546 0.077901108087223641663 -0.31678407293622701246 0.38252291147216815892 0.20802189172017743202 -0.078668026917802738596 0.33114226449472089486 -0.10727017441332237224 0.26628246693473756057 0.13412833645393387738 -0.43368014673425514527;-0.10974605680146130859 0.052834215449894653127 0.058177034837395030786 0.06351897516400403787 0.14729585541897505241 0.20178617324842421099 0.1095495152047723797 -0.12585938886276124005 -0.042728515169954267572 -0.4016778535167820352 -0.26419947736106713654 0.41945035423721821477 -0.43196426466053738924 -0.04459314832162906278 -0.060098010909184304551 -0.32045134836732935835 -0.005216604861683299052 -0.085986617230590187133 0.051735551737014141838 0.021732867938589067042];

% Layer 3
b3 = [-0.56648295077107047035;-0.43882960585261177044];
LW3_2 = [0.7963083811044631144 0.67664949584925038195 -1.4208785668558558957 0.084384107101240543591 0.3229292132049631503 -0.67473194549515935758 -0.80004690065185524173 -0.83485832813351934512;0.80794454364089773879 0.61923268742601444981 -1.4168187214335310475 -0.10883860652216699461 0.23928302691615577213 -0.69863598273717264497 -0.66430938518627635769 -0.52892164913255956815];

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
