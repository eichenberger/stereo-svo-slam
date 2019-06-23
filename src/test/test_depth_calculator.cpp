// 020-TestCase-2.cpp

// main() provided by Catch in file 020-TestCase-1.cpp.
#include <cmath>
#include <opencv2/opencv.hpp>

#include "catch.hpp"

#include "depth_calculator.hpp"

using namespace cv;

static vector<array<float, 2>> keypoints2d;
static vector<array<float, 3>> keypoints3d;
static vector<uint32_t> err;
static float baseline = 45.1932;
static float fx = 680;
static float fy = 680;
static float cx = 357;
static float cy = 225;
static int window_size = 9;
static int search_x = 40;
static int search_y = 1;
static int margin = 40;
static const int split_count = 16;

// Data exported from python
static float sample_data[3][split_count*split_count] =
{{-2.4885798039215685, -2.200583921568627, -2.4885798039215685, -1.2275661591695501, -1.1923811418685122, -2.4885798039215685, -2.4885798039215685, -1.2585973897058824, -1.2461360294117645, -2.49596431372549, -2.49596431372549, -2.49596431372549, -0.6812210294117647, -0.6783563488843813, -2.49596431372549, -0.36553323529411763, -2.0972007843137255, -2.0972007843137255, -2.030740196078431, -1.964279607843137, -1.0338313725490194, -1.0907355363321798, -1.2525264705882353, -0.9695426989619377, -0.6790538363171356, -0.8534938699690402, -1.9864331372549016, -0.6619474588235293, -0.3073802205882353, -0.34753349264705885, -0.35445647058823526, -0.38417461979913914, -1.2325418181818182, -1.21674, -0.8307573529411765, -1.3347501470588234, -1.099155882352941, -1.226964705882353, -0.9684257142857142, -0.7775888823529411, -0.6380216470588235, -0.6171340336134453, -0.592103422459893, -0.6555430748663101, -0.3738408088235294, -0.34725657352941175, -0.2533809926470588, -0.5427614705882352, -1.276043294117647, -1.1829984705882353, -1.2553666666666665, -1.1741370588235294, -1.1479556149732622, -1.1889060784313723, -0.8844370588235293, -1.105662513368984, -0.5221903361344538, -0.5172367519181585, -0.5558521925133689, -0.5226218983957219, -0.2784159777424483, -0.40105527383367134, -1.3070582352941176, -0.19107419117647056, -0.26369846299810246, -0.5192233455882354, -1.1002919607843136, -1.12983, -0.923063725490196, -1.0633694117647057, -1.026446862745098, -1.0412158823529412, -0.45525502941176466, -0.7698351470588235, -0.12322900735294118, -0.3291381512605042, -0.13443164438502675, -0.13009306633291615, -0.263627, -0.6335909411764706, -0.070614375, -0.18729802139037433, -0.24685361344537812, -0.6129143137254901, -0.6202988235294117, -0.5538382352941176, -0.5021466666666666, -0.5021466666666666, -0.30381983193277307, -0.24771673796791444, -0.131172213622291, -0.05997662840746054, -0.07892194852941176, -0.09553709558823528, -0.06507599264705882, -0.2990726470588235, -0.053999227941176474, -0.010113567774936062, -0.09062807486631017, -0.2658423529411764, -0.43568607843137247, -0.45783960784313715, -0.3692254901960784, -0.37660999999999994, -0.15507470588235292, -0.01545595075239398, -0.11892947368421053, -0.055683195548489654, -0.3175339215686274, -0.19938176470588237, -0.02159969117647059, -0.2104585294117647, 0.03876867647058823, 0.028739713831478532, 0.037384080882352934, 0.07476816176470587, 0.0431095707472178, 0.045834888438133874, 0.04372407120743034, -0.12553666666666666, 0.022153529411764704, 0.08307573529411764, 0.11815215686274508, 0.11847322250639386, 0.12342680672268906, 0.05095311764705882, 0.10102009411764706, 0.013750466531440162, 0.2531831932773109, 0.2658423529411764, 0.10906352941176468, 0.10735941176470587, 0.19071299232736574, 0.19384338235294116, 0.24485479876160987, 0.37660999999999994, 0.31458011764705884, 0.22470008403361344, 0.14842864705882353, 0.13750466531440164, 0.142088154158215, 0.12604594320486817, 0.13750466531440164, 0.13956723529411766, 0.7310664705882353, 0.18968959558823528, 0.507948781512605, 0.7775888823529411, 0.7377125294117647, 0.7842349411764705, 0.6055298039215685, 0.6055298039215685, 0.3702804201680672, 0.26125886409736304, 0.29195186974789916, 0.33761978823529415, 0.26125886409736304, 0.24210642857142856, 0.25050529411764705, 0.2907650735294117, 0.8333135294117647, 1.0899536470588234, 1.12983, 0.8174652352941175, 0.8035689304812834, 0.9570324705882352, 0.9570324705882352, 0.8241112941176469, 0.9969088235294116, 0.75322, 0.7785383193277311, 0.9138330882352941, 0.7927798739495798, 0.559050830449827, 0.23614719649561952, 0.34476430147058823, 1.093578770053476, 0.9470633823529411, 0.9913704411764706, 0.974755294117647, 0.7485560990712073, 0.9802936764705883, 0.8108191764705882, 0.7930963529411765, 0.8063884705882353, 0.7842349411764706, 0.7620814117647059, 0.8905718823529412, 0.7642967647058823, 0.3395269181585678, 0.3470719607843137, 0.46730101102941174, 0.6938485411764707, 0.6991653882352942, 0.7532199999999999, 0.6991653882352942, 0.7875579705882352, 0.5821030831643003, 0.7220036631016042, 0.5649150000000001, 0.6722744117647059, 0.6620497058823529, 0.618594705882353, 0.6473753594771242, 0.6773867647058823, 0.69528, 0.3876867647058823, 0.5088388786764706, 2.171045882352941, 2.1415078431372545, 0.7230912, 2.200583921568627, 0.6978361764705883, 0.6978361764705883, 2.0972007843137255, 0.6978361764705883, 2.0750472549019605, 1.2184441176470588, 2.2079684313725485, 0.4553257321652065, 1.912588039215686, 1.9568950980392155, 1.8978190196078428, 0.4541473529411765, 0.8946617647058822, 0.8997741176470588, 0.9023302941176471, 0.8946617647058822, 0.9023302941176471, 0.9023302941176471, 0.8615261437908496, 0.8639876470588236, 0.8566031372549018, 0.871372156862745, 0.5928984055727554, 0.5765069630642955, 1.149768176470588, 0.48460845588235285, 0.6197000794912558, 0.5674711764705881, 0.6159761836441894, 0.9330044117647058, 0.8711084243697478, 0.8433619472616632, 0.8433619472616632, 0.8433619472616632, 0.8711084243697478, 0.8433619472616632, 0.8433619472616632, 0.9009101960784313, 2.8134982352941176, 2.820882745098039, 0.9550632679738561, 2.7618066666666667, 1.1902487165775402, 1.5403218685121107}, {-1.521209019607843, -1.1372145098039215, -1.0338313725490194, -0.4613146712802768, -0.32057460207612454, -0.45783960784313715, -0.2067662745098039, 0.08722952205882353, 0.12461360294117647, 0.43568607843137247, 0.6055298039215685, 0.9599862745098038, 0.35841245798319327, 0.4468901622718052, 1.4990554901960782, 0.3267645588235294, -1.36613431372549, -1.1519835294117646, -0.9452172549019606, -0.7458354901960783, -0.23630431372549016, -0.2345667820069204, -0.046011176470588236, 0.07037003460207612, 0.09535649616368287, 0.262344427244582, 0.7753735294117646, 0.3642040235294118, 0.21738150735294118, 0.2603039705882353, 0.31489659663865543, 0.40200550932568146, -1.2083743315508022, -0.8895494117647058, -0.5898377205882352, -0.6590674999999999, -0.3987635294117647, -0.3016288235294117, -0.08544932773109243, 0.06978361764705882, 0.08639876470588236, 0.21204092436974786, 0.3262610695187166, 0.4017844652406417, 0.3053033272058823, 0.28910355882352934, 0.2824575, 0.6424523529411764, -1.3358578235294116, -1.1098918235294115, -1.0042933333333333, -0.8492186274509803, -0.44105663101604276, -0.30276490196078426, -0.07668529411764706, 0.1208374331550802, 0.12659159663865546, 0.19071299232736574, 0.2598004812834224, 0.3806379144385027, 0.2604536565977742, 0.4102222515212982, 1.5581315686274508, 0.3253799632352941, -0.38375629981024667, -0.7352202573529412, -0.8935256862745097, -0.819680588235294, -0.6719903921568626, -0.31014941176470584, -0.04430705882352941, 0.02953803921568627, 0.09969088235294117, 0.3212261764705882, 0.12738279411764705, 0.40509310924369746, 0.2250597192513369, 0.27432668335419275, 0.45857805882352937, 1.1032457647058822, -0.2533809926470588, -0.47126598930481284, -0.42724663865546214, -0.6867594117647059, -0.6572213725490196, -0.31014941176470584, -0.19938176470588231, 0.10338313725490195, 0.11076764705882351, 0.19636082887700534, 0.18713902476780184, 0.19127681492109036, 0.22430448529411762, 0.23538125, 0.2713807352941176, 1.0342929044117646, -0.8473725, -0.22538808184143222, -0.4380356951871658, -0.804911568627451, -0.6719903921568626, -0.2658423529411764, -0.21415078431372547, -0.036922549019607835, 0.19938176470588231, 0.09119010943912448, 0.3043195356037151, 0.20477046104928454, 1.211059607843137, 0.7798042352941177, 0.3372874852941176, 1.2849047058823528, -0.27691911764705884, -0.3197293163751987, -0.2063047426470588, -0.16199768382352941, -0.1293287122416534, -0.08021105476673428, -0.05246888544891641, -0.02953803921568627, 0.1919972549019608, 0.3710716176470588, 0.3396874509803921, 0.37275721227621483, 0.49687201680672266, 0.43199382352941174, 0.5263678588235294, 0.5271012170385396, -0.6108044537815125, -1.1667525490196078, -0.20960647058823526, -0.20108588235294114, -0.2600631713554987, -0.19384338235294116, -0.09444399380804952, -0.036922549019607835, 0.12405976470588236, 0.16773386554621847, 0.20159711764705882, 0.27730107505070994, 0.3620956186612576, 0.3918882961460446, 0.49043330628803244, 0.5206079411764706, -1.521209019607843, -0.24507341911764707, -0.655111512605042, -0.6180834705882353, -0.5117465294117647, -0.3788253529411764, -0.19938176470588231, 0.10338313725490195, 0.13292117647058824, 0.17188083164300202, 0.19226098739495798, 0.3615456, 0.3323029411764706, 0.4462353781512605, 0.5419094117647059, 0.5233771323529411, -0.945785294117647, -1.0700154705882352, -1.0190623529411764, -0.6247295294117646, -0.44105663101604276, -0.3123647647058823, -0.1727975294117647, -0.006646058823529411, 0.1919972549019608, 0.3455950588235294, 0.38926915966386555, 0.6092220588235294, 0.6646058823529412, 0.6841531141868511, 0.2898812891113892, 0.5005313051470588, -1.196290588235294, -0.9027563235294117, -0.6756826470588235, -0.6590674999999999, -0.25534857585139314, -0.2547655882352941, -0.09304482352941176, 0.09747552941176471, 0.11962905882352941, 0.35002576470588237, 0.36331788235294116, 0.598145294117647, 0.6106066544117646, 0.26728714833759587, 0.3057187058823529, 0.4839161580882353, -0.4864915058823529, -0.4731993882352941, -0.4652241176470588, -0.27381762352941175, -0.2857805294117647, -0.12375419878296146, -0.06646058823529412, 0.017722823529411763, 0.11502794117647058, 0.15337058823529412, 0.27095470588235293, 0.29291888888888884, 0.36297705882352943, 0.4933420588235294, 0.32597336134453775, 0.5254540257352941, -1.3439807843137253, -1.3144427450980392, -0.3961051058823529, -0.7458354901960783, -0.2326120588235294, -0.16103911764705883, -0.21415078431372547, -0.012780882352941177, 0.3323029411764706, 0.2348274117647059, 0.7384509803921568, 0.19655365456821025, 1.144599019607843, 1.3292117647058823, 1.4621329411764705, 0.3115340073529411, -0.4626679411764706, -0.4498870588235294, -0.3783141176470588, -0.27095470588235293, -0.23005588235294117, -0.16103911764705883, -0.07138359477124183, 0.054153071895424836, 0.12061366013071893, 0.13045967320261437, 0.17664524767801856, 0.20710974008207933, 0.5316847058823528, 0.23122746323529408, 0.3808012082670906, 0.38853882352941166, -0.33392393113342894, -0.4549994117647059, -0.35366527310924367, -0.2750093306288033, -0.20854874239350912, -0.1443798985801217, -0.08070214285714286, -0.011458722109533468, 0.05500186612576065, 0.12799816993464053, 0.7827580392156862, 0.8418341176470586, 0.3692254901960784, 1.2701356862745097, 0.598145294117647, 0.9578143598615917}, {5.021466666666666, 5.021466666666666, 5.021466666666666, 2.6584235294117646, 2.6584235294117646, 5.021466666666666, 5.021466666666666, 2.824575, 2.824575, 5.021466666666666, 5.021466666666666, 5.021466666666666, 1.6140428571428571, 1.5583862068965517, 5.021466666666666, 0.941525, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 2.510733333333333, 2.6584235294117646, 3.4764, 2.6584235294117646, 1.9649217391304348, 2.3785894736842104, 5.021466666666666, 1.807728, 0.941525, 0.941525, 1.0760285714285713, 1.1022731707317073, 4.108472727272727, 3.4764, 2.824575, 3.7661, 3.4764, 3.4764, 3.2280857142857142, 2.25966, 2.25966, 2.1520571428571427, 2.0542363636363636, 2.0542363636363636, 1.4122875, 1.12983, 0.941525, 1.88305, 4.51932, 4.51932, 5.021466666666666, 5.021466666666666, 4.108472727272727, 5.021466666666666, 3.4764, 4.108472727272727, 2.1520571428571427, 1.9649217391304348, 2.0542363636363636, 2.0542363636363636, 1.2214378378378377, 1.5583862068965517, 5.021466666666666, 0.941525, 1.4578451612903225, 2.824575, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 2.25966, 3.7661, 0.941525, 2.1520571428571427, 1.0271181818181818, 0.9615574468085106, 1.50644, 3.01288, 0.941525, 2.0542363636363636, 2.1520571428571427, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 2.1520571428571427, 2.0542363636363636, 1.1892947368421052, 1.1022731707317073, 0.941525, 0.941525, 0.941525, 2.824575, 2.824575, 0.9824608695652174, 2.0542363636363636, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 5.021466666666666, 1.0510046511627906, 2.3785894736842104, 1.2214378378378377, 5.021466666666666, 3.01288, 1.12983, 3.7661, 0.941525, 1.2214378378378377, 0.941525, 0.941525, 1.2214378378378377, 1.5583862068965517, 1.1892947368421052, 5.021466666666666, 5.021466666666666, 3.7661, 2.510733333333333, 1.9649217391304348, 2.1520571428571427, 1.50644, 1.807728, 1.5583862068965517, 2.1520571428571427, 5.021466666666666, 1.1587999999999998, 1.1587999999999998, 1.9649217391304348, 3.7661, 2.3785894736842104, 5.021466666666666, 3.01288, 2.1520571428571427, 1.50644, 1.5583862068965517, 1.5583862068965517, 1.5583862068965517, 1.5583862068965517, 1.50644, 5.021466666666666, 0.941525, 3.2280857142857142, 4.51932, 4.51932, 4.51932, 5.021466666666666, 5.021466666666666, 3.2280857142857142, 1.5583862068965517, 1.6140428571428571, 1.807728, 1.5583862068965517, 1.6140428571428571, 1.7382, 1.4122875, 3.4764, 4.51932, 5.021466666666666, 4.51932, 4.108472727272727, 4.51932, 4.51932, 4.51932, 5.021466666666666, 3.01288, 3.2280857142857142, 3.7661, 3.2280857142857142, 2.6584235294117646, 0.9615574468085106, 1.4122875, 4.108472727272727, 3.7661, 3.7661, 3.7661, 2.3785894736842104, 3.7661, 3.01288, 3.01288, 3.01288, 3.01288, 3.01288, 3.01288, 2.824575, 0.9824608695652174, 1.0042933333333333, 1.4122875, 1.807728, 1.807728, 2.1520571428571427, 1.807728, 2.25966, 1.5583862068965517, 2.0542363636363636, 1.50644, 1.7382, 1.7382, 1.7382, 1.673822222222222, 1.7382, 1.7382, 1.0760285714285713, 1.4122875, 5.021466666666666, 5.021466666666666, 1.807728, 5.021466666666666, 1.7382, 1.7382, 5.021466666666666, 1.7382, 5.021466666666666, 3.01288, 5.021466666666666, 0.9615574468085106, 5.021466666666666, 5.021466666666666, 5.021466666666666, 0.941525, 1.7382, 1.7382, 1.7382, 1.7382, 1.7382, 1.7382, 1.673822222222222, 1.673822222222222, 1.673822222222222, 1.673822222222222, 1.1892947368421052, 1.0510046511627906, 2.25966, 0.941525, 1.2214378378378377, 1.1587999999999998, 1.1022731707317073, 1.7382, 1.6140428571428571, 1.5583862068965517, 1.5583862068965517, 1.5583862068965517, 1.6140428571428571, 1.5583862068965517, 1.5583862068965517, 1.673822222222222, 5.021466666666666, 5.021466666666666, 1.673822222222222, 5.021466666666666, 2.0542363636363636, 2.6584235294117646}};

static inline void setup()
{
   if (keypoints2d.size() != 0)
      return;

   auto left = imread("left.png");
   auto right = imread("right.png");
   DepthCalculator depth_calculator(baseline, fx, fy, cx, cy, window_size,
         search_x, search_y, margin);

   depth_calculator.calculate_depth(left, right, split_count, keypoints2d, keypoints3d, err);
}

TEST_CASE( "Depth Calculator points", "[multi-file:2]" ) {
   setup();

   REQUIRE(keypoints2d.size() == split_count*split_count);
   REQUIRE(keypoints3d.size() == split_count*split_count);
   REQUIRE(err.size() == split_count*split_count);

}

TEST_CASE( "Depth Calculator test 3D points", "[multi-file:2]" ) {
   setup();

   for (auto kp: keypoints3d) {
      REQUIRE(kp[2] > 0);
      REQUIRE((abs(kp[0] - kp[1])) > 0);
//      bool match = false;
//      for (int i = 0; i < split_count*split_count; i++) {
//         if (sample_data[0][i] == kp[0] &&
//             sample_data[1][i] == kp[1] &&
//             sample_data[2][i] == kp[2])
//            match = true;
//
//      }
//      REQUIRE(match);
   }
}
