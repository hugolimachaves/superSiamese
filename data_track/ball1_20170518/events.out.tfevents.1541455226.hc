       �K"	  �^/��Abrain.Event:2�_��?q	     U�/�	s!�^/��A"��%
l
PlaceholderPlaceholder*
shape:*
dtype0*&
_output_shapes
:
r
Placeholder_1Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_3Placeholder*
shape:��*
dtype0*(
_output_shapes
:��
M
is_trainingConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
O
is_training_1Const*
dtype0
*
_output_shapes
: *
value	B
 Z 
�
>siamese/scala1/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala1/conv/weights*%
valueB"         `   *
dtype0*
_output_shapes
:
�
=siamese/scala1/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala1/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala1/conv/weights/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:`*

seed*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
seed2	
�
<siamese/scala1/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala1/conv/weights/Initializer/truncated_normal/stddev*&
_output_shapes
:`*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
8siamese/scala1/conv/weights/Initializer/truncated_normalAdd<siamese/scala1/conv/weights/Initializer/truncated_normal/mul=siamese/scala1/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
siamese/scala1/conv/weights
VariableV2*
	container *
shape:`*
dtype0*&
_output_shapes
:`*
shared_name *.
_class$
" loc:@siamese/scala1/conv/weights
�
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(
�
 siamese/scala1/conv/weights/readIdentitysiamese/scala1/conv/weights*&
_output_shapes
:`*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala1/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: 
�
6siamese/scala1/conv/weights/Regularizer/l2_regularizerMul<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: 
�
,siamese/scala1/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala1/conv/biases*
valueB`*���=*
dtype0*
_output_shapes
:`
�
siamese/scala1/conv/biases
VariableV2*-
_class#
!loc:@siamese/scala1/conv/biases*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name 
�
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases
�
siamese/scala1/conv/biases/readIdentitysiamese/scala1/conv/biases*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
_output_shapes
:`
�
siamese/scala1/Conv2DConv2DPlaceholder_2 siamese/scala1/conv/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:;;`*
	dilations
*
T0
�
siamese/scala1/AddAddsiamese/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
�
(siamese/scala1/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala1/bn/beta*
valueB`*    *
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/beta
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *)
_class
loc:@siamese/scala1/bn/beta
�
siamese/scala1/bn/beta/AssignAssignsiamese/scala1/bn/beta(siamese/scala1/bn/beta/Initializer/Const*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(
�
siamese/scala1/bn/beta/readIdentitysiamese/scala1/bn/beta*
T0*)
_class
loc:@siamese/scala1/bn/beta*
_output_shapes
:`
�
)siamese/scala1/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala1/bn/gamma*
valueB`*  �?*
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/gamma
VariableV2*
_output_shapes
:`*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma*
	container *
shape:`*
dtype0
�
siamese/scala1/bn/gamma/AssignAssignsiamese/scala1/bn/gamma)siamese/scala1/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`
�
siamese/scala1/bn/gamma/readIdentitysiamese/scala1/bn/gamma*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
_output_shapes
:`
�
/siamese/scala1/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/moving_mean
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
$siamese/scala1/bn/moving_mean/AssignAssignsiamese/scala1/bn/moving_mean/siamese/scala1/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
"siamese/scala1/bn/moving_mean/readIdentitysiamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
3siamese/scala1/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*  �?*
dtype0*
_output_shapes
:`
�
!siamese/scala1/bn/moving_variance
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
(siamese/scala1/bn/moving_variance/AssignAssign!siamese/scala1/bn/moving_variance3siamese/scala1/bn/moving_variance/Initializer/Const*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
&siamese/scala1/bn/moving_variance/readIdentity!siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
-siamese/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1/moments/meanMeansiamese/scala1/Add-siamese/scala1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
#siamese/scala1/moments/StopGradientStopGradientsiamese/scala1/moments/mean*&
_output_shapes
:`*
T0
�
(siamese/scala1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1/Add#siamese/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
�
1siamese/scala1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala1/moments/varianceMean(siamese/scala1/moments/SquaredDifference1siamese/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
siamese/scala1/moments/SqueezeSqueezesiamese/scala1/moments/mean*
T0*
_output_shapes
:`*
squeeze_dims
 
�
 siamese/scala1/moments/Squeeze_1Squeezesiamese/scala1/moments/variance*
T0*
_output_shapes
:`*
squeeze_dims
 
�
$siamese/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
3siamese/scala1/siamese/scala1/bn/moving_mean/biased
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
:siamese/scala1/siamese/scala1/bn/moving_mean/biased/AssignAssign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zeros*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biased*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Isiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala1/siamese/scala1/bn/moving_mean/local_step
VariableV2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
>siamese/scala1/siamese/scala1/bn/moving_mean/local_step/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepIsiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zeros*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
<siamese/scala1/siamese/scala1/bn/moving_mean/local_step/readIdentity7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readsiamese/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMul@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub$siamese/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
isiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biased@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Lsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepLsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Asiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x$siamese/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/x@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
&siamese/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
7siamese/scala1/siamese/scala1/bn/moving_variance/biased
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`*
dtype0*
_output_shapes
:`
�
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zeros*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Msiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *    
�
;siamese/scala1/siamese/scala1/bn/moving_variance/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/AssignAssign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepMsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala1/siamese/scala1/bn/moving_variance/local_step/readIdentity;siamese/scala1/siamese/scala1/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read siamese/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub&siamese/scala1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
ssiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Rsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepRsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x&siamese/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivGsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
 siamese/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
e
siamese/scala1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala1/cond/switch_tIdentitysiamese/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala1/cond/switch_fIdentitysiamese/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese/scala1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala1/cond/Switch_1Switchsiamese/scala1/moments/Squeezesiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*1
_class'
%#loc:@siamese/scala1/moments/Squeeze
�
siamese/scala1/cond/Switch_2Switch siamese/scala1/moments/Squeeze_1siamese/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese/scala1/cond/MergeMergesiamese/scala1/cond/Switch_3siamese/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese/scala1/cond/Merge_1Mergesiamese/scala1/cond/Switch_4siamese/scala1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
c
siamese/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala1/batchnorm/addAddsiamese/scala1/cond/Merge_1siamese/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
j
siamese/scala1/batchnorm/RsqrtRsqrtsiamese/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/mulMulsiamese/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/mul_1Mulsiamese/scala1/Addsiamese/scala1/batchnorm/mul*&
_output_shapes
:;;`*
T0
�
siamese/scala1/batchnorm/mul_2Mulsiamese/scala1/cond/Mergesiamese/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/subSubsiamese/scala1/bn/beta/readsiamese/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/add_1Addsiamese/scala1/batchnorm/mul_1siamese/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
l
siamese/scala1/ReluRelusiamese/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese/scala1/poll/MaxPoolMaxPoolsiamese/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:`
�
>siamese/scala2/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala2/conv/weights*%
valueB"      0      *
dtype0*
_output_shapes
:
�
=siamese/scala2/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala2/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala2/conv/weights/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
seed2w*
dtype0*'
_output_shapes
:0�*

seed
�
<siamese/scala2/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala2/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
8siamese/scala2/conv/weights/Initializer/truncated_normalAdd<siamese/scala2/conv/weights/Initializer/truncated_normal/mul=siamese/scala2/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
siamese/scala2/conv/weights
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala2/conv/weights*
	container *
shape:0�*
dtype0*'
_output_shapes
:0�
�
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*
validate_shape(*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala2/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
_output_shapes
: 
�
6siamese/scala2/conv/weights/Regularizer/l2_regularizerMul<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
_output_shapes
: 
�
,siamese/scala2/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala2/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala2/conv/biases
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala2/conv/biases*
	container 
�
!siamese/scala2/conv/biases/AssignAssignsiamese/scala2/conv/biases,siamese/scala2/conv/biases/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
�
siamese/scala2/conv/biases/readIdentitysiamese/scala2/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
V
siamese/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala2/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese/scala2/splitSplitsiamese/scala2/split/split_dimsiamese/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
X
siamese/scala2/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
b
 siamese/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/split_1Split siamese/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
\
siamese/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala2/AddAddsiamese/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:�
�
(siamese/scala2/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala2/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/beta
VariableV2*)
_class
loc:@siamese/scala2/bn/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
siamese/scala2/bn/beta/AssignAssignsiamese/scala2/bn/beta(siamese/scala2/bn/beta/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(
�
siamese/scala2/bn/beta/readIdentitysiamese/scala2/bn/beta*
T0*)
_class
loc:@siamese/scala2/bn/beta*
_output_shapes	
:�
�
)siamese/scala2/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala2/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/gamma
VariableV2**
_class 
loc:@siamese/scala2/bn/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
siamese/scala2/bn/gamma/AssignAssignsiamese/scala2/bn/gamma)siamese/scala2/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
_output_shapes	
:�
�
/siamese/scala2/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/moving_mean
VariableV2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
"siamese/scala2/bn/moving_mean/readIdentitysiamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala2/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala2/bn/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
(siamese/scala2/bn/moving_variance/AssignAssign!siamese/scala2/bn/moving_variance3siamese/scala2/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
&siamese/scala2/bn/moving_variance/readIdentity!siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese/scala2/moments/meanMeansiamese/scala2/Add-siamese/scala2/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala2/moments/SquaredDifferenceSquaredDifferencesiamese/scala2/Add#siamese/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
1siamese/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/varianceMean(siamese/scala2/moments/SquaredDifference1siamese/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
siamese/scala2/moments/SqueezeSqueezesiamese/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
 siamese/scala2/moments/Squeeze_1Squeezesiamese/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
$siamese/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�
�
:siamese/scala2/siamese/scala2/bn/moving_mean/biased/AssignAssign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zeros*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Isiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala2/siamese/scala2/bn/moving_mean/local_step
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container 
�
>siamese/scala2/siamese/scala2/bn/moving_mean/local_step/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepIsiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zeros*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
<siamese/scala2/siamese/scala2/bn/moving_mean/local_step/readIdentity7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readsiamese/scala2/moments/Squeeze*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMul@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biased@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Lsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepLsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Asiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x$siamese/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/x@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivAsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
&siamese/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala2/siamese/scala2/bn/moving_variance/biased
VariableV2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala2/siamese/scala2/bn/moving_variance/biased/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala2/siamese/scala2/bn/moving_variance/local_step
VariableV2*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape: *
dtype0
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read siamese/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Rsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivGsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
e
siamese/scala2/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

g
siamese/scala2/cond/switch_tIdentitysiamese/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala2/cond/switch_fIdentitysiamese/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese/scala2/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala2/cond/Switch_1Switchsiamese/scala2/moments/Squeezesiamese/scala2/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala2/cond/Switch_2Switch siamese/scala2/moments/Squeeze_1siamese/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese/scala2/cond/MergeMergesiamese/scala2/cond/Switch_3siamese/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala2/cond/Merge_1Mergesiamese/scala2/cond/Switch_4siamese/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala2/batchnorm/addAddsiamese/scala2/cond/Merge_1siamese/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
k
siamese/scala2/batchnorm/RsqrtRsqrtsiamese/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala2/batchnorm/mulMulsiamese/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/mul_1Mulsiamese/scala2/Addsiamese/scala2/batchnorm/mul*
T0*'
_output_shapes
:�
�
siamese/scala2/batchnorm/mul_2Mulsiamese/scala2/cond/Mergesiamese/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala2/batchnorm/subSubsiamese/scala2/bn/beta/readsiamese/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
siamese/scala2/batchnorm/add_1Addsiamese/scala2/batchnorm/mul_1siamese/scala2/batchnorm/sub*'
_output_shapes
:�*
T0
m
siamese/scala2/ReluRelusiamese/scala2/batchnorm/add_1*
T0*'
_output_shapes
:�
�
siamese/scala2/poll/MaxPoolMaxPoolsiamese/scala2/Relu*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize

�
>siamese/scala3/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala3/conv/weights*%
valueB"         �  *
dtype0*
_output_shapes
:
�
=siamese/scala3/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala3/conv/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:��*

seed*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
seed2�
�
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
siamese/scala3/conv/weights
VariableV2*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights
�
"siamese/scala3/conv/weights/AssignAssignsiamese/scala3/conv/weights8siamese/scala3/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��
�
 siamese/scala3/conv/weights/readIdentitysiamese/scala3/conv/weights*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala3/conv/weights/read*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
6siamese/scala3/conv/weights/Regularizer/l2_regularizerMul<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
_output_shapes
: 
�
,siamese/scala3/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala3/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala3/conv/biases
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala3/conv/biases*
	container *
shape:�
�
!siamese/scala3/conv/biases/AssignAssignsiamese/scala3/conv/biases,siamese/scala3/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala3/conv/biases/readIdentitysiamese/scala3/conv/biases*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
_output_shapes	
:�
�
siamese/scala3/Conv2DConv2Dsiamese/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:

�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese/scala3/AddAddsiamese/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
(siamese/scala3/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala3/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/beta
VariableV2*
shared_name *)
_class
loc:@siamese/scala3/bn/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/beta/AssignAssignsiamese/scala3/bn/beta(siamese/scala3/bn/beta/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta
�
siamese/scala3/bn/beta/readIdentitysiamese/scala3/bn/beta*
_output_shapes	
:�*
T0*)
_class
loc:@siamese/scala3/bn/beta
�
)siamese/scala3/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala3/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/gamma
VariableV2*
shared_name **
_class 
loc:@siamese/scala3/bn/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/gamma/AssignAssignsiamese/scala3/bn/gamma)siamese/scala3/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
siamese/scala3/bn/gamma/readIdentitysiamese/scala3/bn/gamma**
_class 
loc:@siamese/scala3/bn/gamma*
_output_shapes	
:�*
T0
�
/siamese/scala3/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/moving_mean
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
$siamese/scala3/bn/moving_mean/AssignAssignsiamese/scala3/bn/moving_mean/siamese/scala3/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
"siamese/scala3/bn/moving_mean/readIdentitysiamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
3siamese/scala3/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala3/bn/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�
�
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
-siamese/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/meanMeansiamese/scala3/Add-siamese/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala3/moments/StopGradientStopGradientsiamese/scala3/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala3/moments/SquaredDifferenceSquaredDifferencesiamese/scala3/Add#siamese/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
�
1siamese/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/varianceMean(siamese/scala3/moments/SquaredDifference1siamese/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
siamese/scala3/moments/SqueezeSqueezesiamese/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
 siamese/scala3/moments/Squeeze_1Squeezesiamese/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
$siamese/scala3/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala3/siamese/scala3/bn/moving_mean/biased
VariableV2*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zeros*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Isiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zerosConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *    *
dtype0
�
7siamese/scala3/siamese/scala3/bn/moving_mean/local_step
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
�
>siamese/scala3/siamese/scala3/bn/moving_mean/local_step/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepIsiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readsiamese/scala3/moments/Squeeze*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
isiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biased@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Lsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepLsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Asiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x$siamese/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/x@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivAsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
&siamese/scala3/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala3/siamese/scala3/bn/moving_variance/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�
�
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
<siamese/scala3/siamese/scala3/bn/moving_variance/biased/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala3/siamese/scala3/bn/moving_variance/local_step
VariableV2*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Bsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/AssignAssign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepMsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
@siamese/scala3/siamese/scala3/bn/moving_variance/local_step/readIdentity;siamese/scala3/siamese/scala3/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read siamese/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub&siamese/scala3/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Rsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Gsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x&siamese/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivGsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
 siamese/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala3/cond/switch_tIdentitysiamese/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala3/cond/switch_fIdentitysiamese/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala3/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala3/cond/Switch_1Switchsiamese/scala3/moments/Squeezesiamese/scala3/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_2Switch siamese/scala3/moments/Squeeze_1siamese/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala3/batchnorm/addAddsiamese/scala3/cond/Merge_1siamese/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
k
siamese/scala3/batchnorm/RsqrtRsqrtsiamese/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/mulMulsiamese/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/mul_1Mulsiamese/scala3/Addsiamese/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
�
siamese/scala3/batchnorm/mul_2Mulsiamese/scala3/cond/Mergesiamese/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/subSubsiamese/scala3/bn/beta/readsiamese/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/add_1Addsiamese/scala3/batchnorm/mul_1siamese/scala3/batchnorm/sub*'
_output_shapes
:

�*
T0
m
siamese/scala3/ReluRelusiamese/scala3/batchnorm/add_1*'
_output_shapes
:

�*
T0
�
>siamese/scala4/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala4/conv/weights*%
valueB"      �   �  *
dtype0*
_output_shapes
:
�
=siamese/scala4/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala4/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala4/conv/weights/Initializer/truncated_normal/shape*(
_output_shapes
:��*

seed*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
seed2�*
dtype0
�
<siamese/scala4/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala4/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
8siamese/scala4/conv/weights/Initializer/truncated_normalAdd<siamese/scala4/conv/weights/Initializer/truncated_normal/mul=siamese/scala4/conv/weights/Initializer/truncated_normal/mean*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��*
T0
�
siamese/scala4/conv/weights
VariableV2*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala4/conv/weights
�
"siamese/scala4/conv/weights/AssignAssignsiamese/scala4/conv/weights8siamese/scala4/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��
�
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��*
T0
�
<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala4/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: 
�
6siamese/scala4/conv/weights/Regularizer/l2_regularizerMul<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
,siamese/scala4/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala4/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala4/conv/biases
VariableV2*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala4/conv/biases*
	container *
shape:�*
dtype0
�
!siamese/scala4/conv/biases/AssignAssignsiamese/scala4/conv/biases,siamese/scala4/conv/biases/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(
�
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
V
siamese/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/splitSplitsiamese/scala4/split/split_dimsiamese/scala3/Relu*:
_output_shapes(
&:

�:

�*
	num_split*
T0
X
siamese/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala4/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese/scala4/split_1Split siamese/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala4/Conv2DConv2Dsiamese/scala4/splitsiamese/scala4/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
\
siamese/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/concatConcatV2siamese/scala4/Conv2Dsiamese/scala4/Conv2D_1siamese/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese/scala4/AddAddsiamese/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
(siamese/scala4/bn/beta/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*)
_class
loc:@siamese/scala4/bn/beta*
valueB�*    
�
siamese/scala4/bn/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala4/bn/beta*
	container *
shape:�
�
siamese/scala4/bn/beta/AssignAssignsiamese/scala4/bn/beta(siamese/scala4/bn/beta/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta
�
siamese/scala4/bn/beta/readIdentitysiamese/scala4/bn/beta*
T0*)
_class
loc:@siamese/scala4/bn/beta*
_output_shapes	
:�
�
)siamese/scala4/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala4/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala4/bn/gamma*
	container *
shape:�
�
siamese/scala4/bn/gamma/AssignAssignsiamese/scala4/bn/gamma)siamese/scala4/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
siamese/scala4/bn/gamma/readIdentitysiamese/scala4/bn/gamma**
_class 
loc:@siamese/scala4/bn/gamma*
_output_shapes	
:�*
T0
�
/siamese/scala4/bn/moving_mean/Initializer/ConstConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0
�
siamese/scala4/bn/moving_mean
VariableV2*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�*
dtype0
�
$siamese/scala4/bn/moving_mean/AssignAssignsiamese/scala4/bn/moving_mean/siamese/scala4/bn/moving_mean/Initializer/Const*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
"siamese/scala4/bn/moving_mean/readIdentitysiamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
3siamese/scala4/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala4/bn/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape:�
�
(siamese/scala4/bn/moving_variance/AssignAssign!siamese/scala4/bn/moving_variance3siamese/scala4/bn/moving_variance/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
&siamese/scala4/bn/moving_variance/readIdentity!siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
-siamese/scala4/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese/scala4/moments/meanMeansiamese/scala4/Add-siamese/scala4/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala4/moments/StopGradientStopGradientsiamese/scala4/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala4/moments/SquaredDifferenceSquaredDifferencesiamese/scala4/Add#siamese/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
1siamese/scala4/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala4/moments/varianceMean(siamese/scala4/moments/SquaredDifference1siamese/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
siamese/scala4/moments/SqueezeSqueezesiamese/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
 siamese/scala4/moments/Squeeze_1Squeezesiamese/scala4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
$siamese/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala4/siamese/scala4/bn/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�
�
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
�
8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biased*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Isiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala4/siamese/scala4/bn/moving_mean/local_step
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
�
>siamese/scala4/siamese/scala4/bn/moving_mean/local_step/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepIsiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zeros*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(
�
<siamese/scala4/siamese/scala4/bn/moving_mean/local_step/readIdentity7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readsiamese/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMul@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub$siamese/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biased@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Lsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Fsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepLsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Asiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x$siamese/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/x@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivAsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
siamese/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
&siamese/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala4/siamese/scala4/bn/moving_variance/biased
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
<siamese/scala4/siamese/scala4/bn/moving_variance/biased/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala4/siamese/scala4/bn/moving_variance/local_step
VariableV2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Bsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/AssignAssign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepMsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala4/siamese/scala4/bn/moving_variance/local_step/readIdentity;siamese/scala4/siamese/scala4/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read siamese/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub&siamese/scala4/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
ssiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Rsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepRsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x&siamese/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivGsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
 siamese/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala4/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

g
siamese/scala4/cond/switch_tIdentitysiamese/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala4/cond/switch_fIdentitysiamese/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese/scala4/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala4/cond/Switch_1Switchsiamese/scala4/moments/Squeezesiamese/scala4/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_2Switch siamese/scala4/moments/Squeeze_1siamese/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala4/cond/MergeMergesiamese/scala4/cond/Switch_3siamese/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala4/cond/Merge_1Mergesiamese/scala4/cond/Switch_4siamese/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala4/batchnorm/addAddsiamese/scala4/cond/Merge_1siamese/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
k
siamese/scala4/batchnorm/RsqrtRsqrtsiamese/scala4/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese/scala4/batchnorm/mulMulsiamese/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/mul_1Mulsiamese/scala4/Addsiamese/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
siamese/scala4/batchnorm/mul_2Mulsiamese/scala4/cond/Mergesiamese/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/subSubsiamese/scala4/bn/beta/readsiamese/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/add_1Addsiamese/scala4/batchnorm/mul_1siamese/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
m
siamese/scala4/ReluRelusiamese/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
�
>siamese/scala5/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala5/conv/weights*%
valueB"      �      *
dtype0*
_output_shapes
:
�
=siamese/scala5/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala5/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala5/conv/weights/Initializer/truncated_normal/shape*(
_output_shapes
:��*

seed*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
seed2�*
dtype0
�
<siamese/scala5/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala5/conv/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
8siamese/scala5/conv/weights/Initializer/truncated_normalAdd<siamese/scala5/conv/weights/Initializer/truncated_normal/mul=siamese/scala5/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
siamese/scala5/conv/weights
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights*
	container *
shape:��
�
"siamese/scala5/conv/weights/AssignAssignsiamese/scala5/conv/weights8siamese/scala5/conv/weights/Initializer/truncated_normal*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(
�
 siamese/scala5/conv/weights/readIdentitysiamese/scala5/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *o:*
dtype0
�
=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala5/conv/weights/read*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: *
T0
�
6siamese/scala5/conv/weights/Regularizer/l2_regularizerMul<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: 
�
,siamese/scala5/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala5/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala5/conv/biases
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases*
	container 
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*-
_class#
!loc:@siamese/scala5/conv/biases*
_output_shapes	
:�*
T0
V
siamese/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/splitSplitsiamese/scala5/split/split_dimsiamese/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
X
siamese/scala5/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
b
 siamese/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/split_1Split siamese/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala5/Conv2DConv2Dsiamese/scala5/splitsiamese/scala5/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

\
siamese/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/concatConcatV2siamese/scala5/Conv2Dsiamese/scala5/Conv2D_1siamese/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala5/AddAddsiamese/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
�
siamese/scala1_1/Conv2DConv2DPlaceholder_3 siamese/scala1/conv/weights/read*&
_output_shapes
:{{`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala1_1/AddAddsiamese/scala1_1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese/scala1_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1_1/moments/meanMeansiamese/scala1_1/Add/siamese/scala1_1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese/scala1_1/moments/StopGradientStopGradientsiamese/scala1_1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese/scala1_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1_1/Add%siamese/scala1_1/moments/StopGradient*
T0*&
_output_shapes
:{{`
�
3siamese/scala1_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala1_1/moments/varianceMean*siamese/scala1_1/moments/SquaredDifference3siamese/scala1_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese/scala1_1/moments/SqueezeSqueezesiamese/scala1_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
�
"siamese/scala1_1/moments/Squeeze_1Squeeze!siamese/scala1_1/moments/variance*
T0*
_output_shapes
:`*
squeeze_dims
 
�
&siamese/scala1_1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese/scala1_1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
�
Nsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese/scala1_1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
(siamese/scala1_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese/scala1_1/moments/Squeeze_1*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese/scala1_1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
g
siamese/scala1_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala1_1/cond/switch_tIdentitysiamese/scala1_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala1_1/cond/switch_fIdentitysiamese/scala1_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala1_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala1_1/cond/Switch_1Switch siamese/scala1_1/moments/Squeezesiamese/scala1_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala1_1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_2Switch"siamese/scala1_1/moments/Squeeze_1siamese/scala1_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala1_1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese/scala1_1/cond/MergeMergesiamese/scala1_1/cond/Switch_3 siamese/scala1_1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
�
siamese/scala1_1/cond/Merge_1Mergesiamese/scala1_1/cond/Switch_4 siamese/scala1_1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese/scala1_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese/scala1_1/batchnorm/addAddsiamese/scala1_1/cond/Merge_1 siamese/scala1_1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese/scala1_1/batchnorm/RsqrtRsqrtsiamese/scala1_1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese/scala1_1/batchnorm/mulMul siamese/scala1_1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese/scala1_1/batchnorm/mul_1Mulsiamese/scala1_1/Addsiamese/scala1_1/batchnorm/mul*
T0*&
_output_shapes
:{{`
�
 siamese/scala1_1/batchnorm/mul_2Mulsiamese/scala1_1/cond/Mergesiamese/scala1_1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese/scala1_1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese/scala1_1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese/scala1_1/batchnorm/add_1Add siamese/scala1_1/batchnorm/mul_1siamese/scala1_1/batchnorm/sub*
T0*&
_output_shapes
:{{`
p
siamese/scala1_1/ReluRelu siamese/scala1_1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
�
siamese/scala1_1/poll/MaxPoolMaxPoolsiamese/scala1_1/Relu*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
X
siamese/scala2_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese/scala2_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/splitSplit siamese/scala2_1/split/split_dimsiamese/scala1_1/poll/MaxPool*
T0*8
_output_shapes&
$:==0:==0*
	num_split
Z
siamese/scala2_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala2_1/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala2_1/split_1Split"siamese/scala2_1/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2_1/Conv2DConv2Dsiamese/scala2_1/splitsiamese/scala2_1/split_1*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese/scala2_1/Conv2D_1Conv2Dsiamese/scala2_1/split:1siamese/scala2_1/split_1:1*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese/scala2_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese/scala2_1/concatConcatV2siamese/scala2_1/Conv2Dsiamese/scala2_1/Conv2D_1siamese/scala2_1/concat/axis*
N*'
_output_shapes
:99�*

Tidx0*
T0
�
siamese/scala2_1/AddAddsiamese/scala2_1/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:99�*
T0
�
/siamese/scala2_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2_1/moments/meanMeansiamese/scala2_1/Add/siamese/scala2_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese/scala2_1/moments/StopGradientStopGradientsiamese/scala2_1/moments/mean*
T0*'
_output_shapes
:�
�
*siamese/scala2_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala2_1/Add%siamese/scala2_1/moments/StopGradient*
T0*'
_output_shapes
:99�
�
3siamese/scala2_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala2_1/moments/varianceMean*siamese/scala2_1/moments/SquaredDifference3siamese/scala2_1/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala2_1/moments/SqueezeSqueezesiamese/scala2_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese/scala2_1/moments/Squeeze_1Squeeze!siamese/scala2_1/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese/scala2_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese/scala2_1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala2_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese/scala2_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese/scala2_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese/scala2_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese/scala2_1/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
g
siamese/scala2_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala2_1/cond/switch_tIdentitysiamese/scala2_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala2_1/cond/switch_fIdentitysiamese/scala2_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala2_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/Switch_2Switch"siamese/scala2_1/moments/Squeeze_1siamese/scala2_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala2_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2_1/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese/scala2_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/MergeMergesiamese/scala2_1/cond/Switch_3 siamese/scala2_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala2_1/cond/Merge_1Mergesiamese/scala2_1/cond/Switch_4 siamese/scala2_1/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese/scala2_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/batchnorm/addAddsiamese/scala2_1/cond/Merge_1 siamese/scala2_1/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese/scala2_1/batchnorm/RsqrtRsqrtsiamese/scala2_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala2_1/batchnorm/mulMul siamese/scala2_1/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese/scala2_1/batchnorm/mul_1Mulsiamese/scala2_1/Addsiamese/scala2_1/batchnorm/mul*
T0*'
_output_shapes
:99�
�
 siamese/scala2_1/batchnorm/mul_2Mulsiamese/scala2_1/cond/Mergesiamese/scala2_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala2_1/batchnorm/subSubsiamese/scala2/bn/beta/read siamese/scala2_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese/scala2_1/batchnorm/add_1Add siamese/scala2_1/batchnorm/mul_1siamese/scala2_1/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese/scala2_1/ReluRelu siamese/scala2_1/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese/scala2_1/poll/MaxPoolMaxPoolsiamese/scala2_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
siamese/scala3_1/Conv2DConv2Dsiamese/scala2_1/poll/MaxPool siamese/scala3/conv/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese/scala3_1/AddAddsiamese/scala3_1/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese/scala3_1/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese/scala3_1/moments/meanMeansiamese/scala3_1/Add/siamese/scala3_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese/scala3_1/moments/StopGradientStopGradientsiamese/scala3_1/moments/mean*
T0*'
_output_shapes
:�
�
*siamese/scala3_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala3_1/Add%siamese/scala3_1/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese/scala3_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala3_1/moments/varianceMean*siamese/scala3_1/moments/SquaredDifference3siamese/scala3_1/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala3_1/moments/SqueezeSqueezesiamese/scala3_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese/scala3_1/moments/Squeeze_1Squeeze!siamese/scala3_1/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese/scala3_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese/scala3_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese/scala3_1/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Nsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese/scala3_1/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala3_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese/scala3_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
"siamese/scala3_1/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
g
siamese/scala3_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala3_1/cond/switch_tIdentitysiamese/scala3_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala3_1/cond/switch_fIdentitysiamese/scala3_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala3_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala3_1/cond/Switch_1Switch siamese/scala3_1/moments/Squeezesiamese/scala3_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala3_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_2Switch"siamese/scala3_1/moments/Squeeze_1siamese/scala3_1/cond/pred_id*5
_class+
)'loc:@siamese/scala3_1/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese/scala3_1/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese/scala3_1/cond/MergeMergesiamese/scala3_1/cond/Switch_3 siamese/scala3_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3_1/cond/Merge_1Mergesiamese/scala3_1/cond/Switch_4 siamese/scala3_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala3_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala3_1/batchnorm/addAddsiamese/scala3_1/cond/Merge_1 siamese/scala3_1/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese/scala3_1/batchnorm/RsqrtRsqrtsiamese/scala3_1/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese/scala3_1/batchnorm/mulMul siamese/scala3_1/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese/scala3_1/batchnorm/mul_1Mulsiamese/scala3_1/Addsiamese/scala3_1/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese/scala3_1/batchnorm/mul_2Mulsiamese/scala3_1/cond/Mergesiamese/scala3_1/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala3_1/batchnorm/subSubsiamese/scala3/bn/beta/read siamese/scala3_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese/scala3_1/batchnorm/add_1Add siamese/scala3_1/batchnorm/mul_1siamese/scala3_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala3_1/ReluRelu siamese/scala3_1/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese/scala4_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala4_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/splitSplit siamese/scala4_1/split/split_dimsiamese/scala3_1/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese/scala4_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala4_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/split_1Split"siamese/scala4_1/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala4_1/Conv2DConv2Dsiamese/scala4_1/splitsiamese/scala4_1/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides

�
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese/scala4_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/concatConcatV2siamese/scala4_1/Conv2Dsiamese/scala4_1/Conv2D_1siamese/scala4_1/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese/scala4_1/AddAddsiamese/scala4_1/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese/scala4_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4_1/moments/meanMeansiamese/scala4_1/Add/siamese/scala4_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese/scala4_1/moments/StopGradientStopGradientsiamese/scala4_1/moments/mean*
T0*'
_output_shapes
:�
�
*siamese/scala4_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala4_1/Add%siamese/scala4_1/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese/scala4_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala4_1/moments/varianceMean*siamese/scala4_1/moments/SquaredDifference3siamese/scala4_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese/scala4_1/moments/SqueezeSqueezesiamese/scala4_1/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese/scala4_1/moments/Squeeze_1Squeeze!siamese/scala4_1/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese/scala4_1/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese/scala4_1/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese/scala4_1/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
ksiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese/scala4_1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
 siamese/scala4_1/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala4_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese/scala4_1/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese/scala4_1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese/scala4_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
g
siamese/scala4_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala4_1/cond/switch_tIdentitysiamese/scala4_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala4_1/cond/switch_fIdentitysiamese/scala4_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala4_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala4_1/cond/Switch_1Switch siamese/scala4_1/moments/Squeezesiamese/scala4_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala4_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_2Switch"siamese/scala4_1/moments/Squeeze_1siamese/scala4_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala4_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4_1/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
siamese/scala4_1/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala4_1/cond/Merge_1Mergesiamese/scala4_1/cond/Switch_4 siamese/scala4_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala4_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/batchnorm/addAddsiamese/scala4_1/cond/Merge_1 siamese/scala4_1/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese/scala4_1/batchnorm/RsqrtRsqrtsiamese/scala4_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala4_1/batchnorm/mulMul siamese/scala4_1/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/batchnorm/mul_1Mulsiamese/scala4_1/Addsiamese/scala4_1/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese/scala4_1/batchnorm/mul_2Mulsiamese/scala4_1/cond/Mergesiamese/scala4_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala4_1/batchnorm/subSubsiamese/scala4/bn/beta/read siamese/scala4_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese/scala4_1/batchnorm/add_1Add siamese/scala4_1/batchnorm/mul_1siamese/scala4_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala4_1/ReluRelu siamese/scala4_1/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese/scala5_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese/scala5_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/splitSplit siamese/scala5_1/split/split_dimsiamese/scala4_1/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese/scala5_1/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese/scala5_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/split_1Split"siamese/scala5_1/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese/scala5_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/concatConcatV2siamese/scala5_1/Conv2Dsiamese/scala5_1/Conv2D_1siamese/scala5_1/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese/scala5_1/AddAddsiamese/scala5_1/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
m
score/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose	Transposesiamese/scala5/Addscore/transpose/perm*'
_output_shapes
:�*
Tperm0*
T0
M
score/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
W
score/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
score/splitSplitscore/split/split_dimscore/transpose*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split*
T0
O
score/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
score/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split*
T0
�
score/Conv2DConv2Dscore/split_1score/split*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC
�
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations

�
score/Conv2D_3Conv2Dscore/split_1:3score/split:3*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_4Conv2Dscore/split_1:4score/split:4*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_6Conv2Dscore/split_1:6score/split:6*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
score/Conv2D_7Conv2Dscore/split_1:7score/split:7*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
S
score/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
score/concatConcatV2score/Conv2Dscore/Conv2D_1score/Conv2D_2score/Conv2D_3score/Conv2D_4score/Conv2D_5score/Conv2D_6score/Conv2D_7score/concat/axis*
N*&
_output_shapes
:*

Tidx0*
T0
o
score/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose_1	Transposescore/concatscore/transpose_1/perm*&
_output_shapes
:*
Tperm0*
T0
�
 adjust/weights/Initializer/ConstConst*!
_class
loc:@adjust/weights*%
valueB*o�:*
dtype0*&
_output_shapes
:
�
adjust/weights
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *!
_class
loc:@adjust/weights*
	container *
shape:
�
adjust/weights/AssignAssignadjust/weights adjust/weights/Initializer/Const*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights
�
adjust/weights/readIdentityadjust/weights*
T0*!
_class
loc:@adjust/weights*&
_output_shapes
:
�
/adjust/weights/Regularizer/l2_regularizer/scaleConst*!
_class
loc:@adjust/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
0adjust/weights/Regularizer/l2_regularizer/L2LossL2Lossadjust/weights/read*
T0*!
_class
loc:@adjust/weights*
_output_shapes
: 
�
)adjust/weights/Regularizer/l2_regularizerMul/adjust/weights/Regularizer/l2_regularizer/scale0adjust/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*!
_class
loc:@adjust/weights
�
adjust/biases/Initializer/ConstConst* 
_class
loc:@adjust/biases*
valueB*    *
dtype0*
_output_shapes
:
�
adjust/biases
VariableV2* 
_class
loc:@adjust/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
adjust/biases/AssignAssignadjust/biasesadjust/biases/Initializer/Const*
use_locking(*
T0* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:
t
adjust/biases/readIdentityadjust/biases* 
_class
loc:@adjust/biases*
_output_shapes
:*
T0
�
.adjust/biases/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: * 
_class
loc:@adjust/biases*
valueB
 *o:*
dtype0
�
/adjust/biases/Regularizer/l2_regularizer/L2LossL2Lossadjust/biases/read*
_output_shapes
: *
T0* 
_class
loc:@adjust/biases
�
(adjust/biases/Regularizer/l2_regularizerMul.adjust/biases/Regularizer/l2_regularizer/scale/adjust/biases/Regularizer/l2_regularizer/L2Loss*
T0* 
_class
loc:@adjust/biases*
_output_shapes
: 
�
adjust/Conv2DConv2Dscore/transpose_1adjust/weights/read*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
e

adjust/AddAddadjust/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:,*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights
�
save/SaveV2/shape_and_slicesConst*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesadjust/biasesadjust/weightssiamese/scala1/bn/betasiamese/scala1/bn/gammasiamese/scala1/bn/moving_mean!siamese/scala1/bn/moving_variancesiamese/scala1/conv/biasessiamese/scala1/conv/weights3siamese/scala1/siamese/scala1/bn/moving_mean/biased7siamese/scala1/siamese/scala1/bn/moving_mean/local_step7siamese/scala1/siamese/scala1/bn/moving_variance/biased;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsiamese/scala2/bn/betasiamese/scala2/bn/gammasiamese/scala2/bn/moving_mean!siamese/scala2/bn/moving_variancesiamese/scala2/conv/biasessiamese/scala2/conv/weights3siamese/scala2/siamese/scala2/bn/moving_mean/biased7siamese/scala2/siamese/scala2/bn/moving_mean/local_step7siamese/scala2/siamese/scala2/bn/moving_variance/biased;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsiamese/scala3/bn/betasiamese/scala3/bn/gammasiamese/scala3/bn/moving_mean!siamese/scala3/bn/moving_variancesiamese/scala3/conv/biasessiamese/scala3/conv/weights3siamese/scala3/siamese/scala3/bn/moving_mean/biased7siamese/scala3/siamese/scala3/bn/moving_mean/local_step7siamese/scala3/siamese/scala3/bn/moving_variance/biased;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsiamese/scala4/bn/betasiamese/scala4/bn/gammasiamese/scala4/bn/moving_mean!siamese/scala4/bn/moving_variancesiamese/scala4/conv/biasessiamese/scala4/conv/weights3siamese/scala4/siamese/scala4/bn/moving_mean/biased7siamese/scala4/siamese/scala4/bn/moving_mean/local_step7siamese/scala4/siamese/scala4/bn/moving_variance/biased;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsiamese/scala5/conv/biasessiamese/scala5/conv/weights*:
dtypes0
.2,
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,
�
save/AssignAssignadjust/biasessave/RestoreV2* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_1Assignadjust/weightssave/RestoreV2:1*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(
�
save/Assign_2Assignsiamese/scala1/bn/betasave/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`
�
save/Assign_3Assignsiamese/scala1/bn/gammasave/RestoreV2:3*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save/Assign_4Assignsiamese/scala1/bn/moving_meansave/RestoreV2:4*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2:5*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
save/Assign_6Assignsiamese/scala1/conv/biasessave/RestoreV2:6*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`
�
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2:7*
validate_shape(*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
save/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave/RestoreV2:8*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave/RestoreV2:9*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(
�
save/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave/RestoreV2:10*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave/RestoreV2:11*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_12Assignsiamese/scala2/bn/betasave/RestoreV2:12*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save/Assign_13Assignsiamese/scala2/bn/gammasave/RestoreV2:13*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(
�
save/Assign_14Assignsiamese/scala2/bn/moving_meansave/RestoreV2:14*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2:16*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2:17*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�*
use_locking(*
T0
�
save/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave/RestoreV2:18*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave/RestoreV2:19*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave/RestoreV2:20*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave/RestoreV2:21*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/Assign_22Assignsiamese/scala3/bn/betasave/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2:23**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2:24*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(
�
save/Assign_25Assign!siamese/scala3/bn/moving_variancesave/RestoreV2:25*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2:26*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(
�
save/Assign_27Assignsiamese/scala3/conv/weightssave/RestoreV2:27*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
save/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave/RestoreV2:28*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave/RestoreV2:29*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(
�
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2:30*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2:31*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2:32*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_33Assignsiamese/scala4/bn/gammasave/RestoreV2:33*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(
�
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2:34*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_35Assign!siamese/scala4/bn/moving_variancesave/RestoreV2:35*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave/RestoreV2:39*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave/RestoreV2:40*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave/RestoreV2:41*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(
�
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2:42*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(
�
save/Assign_43Assignsiamese/scala5/conv/weightssave/RestoreV2:43*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:;;`*
	dilations
*
T0
�
siamese_1/scala1/AddAddsiamese_1/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
�
/siamese_1/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala1/moments/meanMeansiamese_1/scala1/Add/siamese_1/scala1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
%siamese_1/scala1/moments/StopGradientStopGradientsiamese_1/scala1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese_1/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
3siamese_1/scala1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_1/scala1/moments/varianceMean*siamese_1/scala1/moments/SquaredDifference3siamese_1/scala1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala1/moments/SqueezeSqueezesiamese_1/scala1/moments/mean*
T0*
_output_shapes
:`*
squeeze_dims
 
�
"siamese_1/scala1/moments/Squeeze_1Squeeze!siamese_1/scala1/moments/variance*
T0*
_output_shapes
:`*
squeeze_dims
 
�
&siamese_1/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_1/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_1/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
ksiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Nsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_1/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
 siamese_1/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
(siamese_1/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_1/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_1/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_1/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_1/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
c
siamese_1/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala1/cond/switch_tIdentitysiamese_1/scala1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_1/scala1/cond/switch_fIdentitysiamese_1/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala1/cond/Switch_1Switch siamese_1/scala1/moments/Squeezesiamese_1/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_2Switch"siamese_1/scala1/moments/Squeeze_1siamese_1/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_1/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_1/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/MergeMergesiamese_1/scala1/cond/Switch_3 siamese_1/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_1/scala1/cond/Merge_1Mergesiamese_1/scala1/cond/Switch_4 siamese_1/scala1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
e
 siamese_1/scala1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_1/scala1/batchnorm/addAddsiamese_1/scala1/cond/Merge_1 siamese_1/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_1/scala1/batchnorm/RsqrtRsqrtsiamese_1/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_1/scala1/batchnorm/mulMul siamese_1/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese_1/scala1/batchnorm/mul_1Mulsiamese_1/scala1/Addsiamese_1/scala1/batchnorm/mul*&
_output_shapes
:;;`*
T0
�
 siamese_1/scala1/batchnorm/mul_2Mulsiamese_1/scala1/cond/Mergesiamese_1/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_1/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_1/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese_1/scala1/batchnorm/add_1Add siamese_1/scala1/batchnorm/mul_1siamese_1/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
p
siamese_1/scala1/ReluRelu siamese_1/scala1/batchnorm/add_1*
T0*&
_output_shapes
:;;`
�
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:`
X
siamese_1/scala2/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_1/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/splitSplit siamese_1/scala2/split/split_dimsiamese_1/scala1/poll/MaxPool*8
_output_shapes&
$:0:0*
	num_split*
T0
Z
siamese_1/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_1/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/split_1Split"siamese_1/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_1/scala2/Conv2DConv2Dsiamese_1/scala2/splitsiamese_1/scala2/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_1/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_1/scala2/AddAddsiamese_1/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_1/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala2/moments/meanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala2/Add%siamese_1/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_1/scala2/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_1/scala2/moments/varianceMean*siamese_1/scala2/moments/SquaredDifference3siamese_1/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala2/moments/SqueezeSqueezesiamese_1/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_1/scala2/moments/Squeeze_1Squeeze!siamese_1/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_1/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_1/scala2/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_1/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
(siamese_1/scala2/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_1/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_1/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_1/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_1/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala2/cond/switch_tIdentitysiamese_1/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala2/cond/switch_fIdentitysiamese_1/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala2/cond/Switch_1Switch siamese_1/scala2/moments/Squeezesiamese_1/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_2Switch"siamese_1/scala2/moments/Squeeze_1siamese_1/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_1/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_1/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_1/scala2/cond/MergeMergesiamese_1/scala2/cond/Switch_3 siamese_1/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala2/cond/Merge_1Mergesiamese_1/scala2/cond/Switch_4 siamese_1/scala2/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_1/scala2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_1/scala2/batchnorm/addAddsiamese_1/scala2/cond/Merge_1 siamese_1/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala2/batchnorm/RsqrtRsqrtsiamese_1/scala2/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_1/scala2/batchnorm/mulMul siamese_1/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_1/scala2/batchnorm/mul_1Mulsiamese_1/scala2/Addsiamese_1/scala2/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_1/scala2/batchnorm/mul_2Mulsiamese_1/scala2/cond/Mergesiamese_1/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_1/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_1/scala2/batchnorm/add_1Add siamese_1/scala2/batchnorm/mul_1siamese_1/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_1/scala2/ReluRelu siamese_1/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese_1/scala2/poll/MaxPoolMaxPoolsiamese_1/scala2/Relu*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides

�
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
_output_shapes
:

�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala3/AddAddsiamese_1/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
/siamese_1/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala3/moments/meanMeansiamese_1/scala3/Add/siamese_1/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala3/moments/StopGradientStopGradientsiamese_1/scala3/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_1/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala3/Add%siamese_1/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
3siamese_1/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala3/moments/varianceMean*siamese_1/scala3/moments/SquaredDifference3siamese_1/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala3/moments/Squeeze_1Squeeze!siamese_1/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_1/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_1/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_1/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_1/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_1/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_1/scala3/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
"siamese_1/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
c
siamese_1/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala3/cond/switch_tIdentitysiamese_1/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_1/scala3/cond/switch_fIdentitysiamese_1/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala3/cond/Switch_1Switch siamese_1/scala3/moments/Squeezesiamese_1/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_2Switch"siamese_1/scala3/moments/Squeeze_1siamese_1/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_1/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_1/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/MergeMergesiamese_1/scala3/cond/Switch_3 siamese_1/scala3/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_1/scala3/cond/Merge_1Mergesiamese_1/scala3/cond/Switch_4 siamese_1/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala3/batchnorm/addAddsiamese_1/scala3/cond/Merge_1 siamese_1/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala3/batchnorm/RsqrtRsqrtsiamese_1/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala3/batchnorm/mulMul siamese_1/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_1/scala3/batchnorm/mul_1Mulsiamese_1/scala3/Addsiamese_1/scala3/batchnorm/mul*
T0*'
_output_shapes
:

�
�
 siamese_1/scala3/batchnorm/mul_2Mulsiamese_1/scala3/cond/Mergesiamese_1/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_1/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala3/batchnorm/add_1Add siamese_1/scala3/batchnorm/mul_1siamese_1/scala3/batchnorm/sub*
T0*'
_output_shapes
:

�
q
siamese_1/scala3/ReluRelu siamese_1/scala3/batchnorm/add_1*
T0*'
_output_shapes
:

�
X
siamese_1/scala4/ConstConst*
_output_shapes
: *
value	B :*
dtype0
b
 siamese_1/scala4/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_1/scala4/splitSplit siamese_1/scala4/split/split_dimsiamese_1/scala3/Relu*
T0*:
_output_shapes(
&:

�:

�*
	num_split
Z
siamese_1/scala4/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_1/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/split_1Split"siamese_1/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_1/scala4/Conv2DConv2Dsiamese_1/scala4/splitsiamese_1/scala4/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
^
siamese_1/scala4/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_1/scala4/concatConcatV2siamese_1/scala4/Conv2Dsiamese_1/scala4/Conv2D_1siamese_1/scala4/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_1/scala4/AddAddsiamese_1/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_1/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala4/moments/meanMeansiamese_1/scala4/Add/siamese_1/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala4/moments/StopGradientStopGradientsiamese_1/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala4/Add%siamese_1/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_1/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala4/moments/varianceMean*siamese_1/scala4/moments/SquaredDifference3siamese_1/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_1/scala4/moments/SqueezeSqueezesiamese_1/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala4/moments/Squeeze_1Squeeze!siamese_1/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_1/scala4/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_1/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_1/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_1/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_1/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_1/scala4/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_1/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
c
siamese_1/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala4/cond/switch_tIdentitysiamese_1/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala4/cond/switch_fIdentitysiamese_1/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala4/cond/Switch_1Switch siamese_1/scala4/moments/Squeezesiamese_1/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_2Switch"siamese_1/scala4/moments/Squeeze_1siamese_1/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_1/scala4/cond/pred_id*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_1/scala4/cond/MergeMergesiamese_1/scala4/cond/Switch_3 siamese_1/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala4/cond/Merge_1Mergesiamese_1/scala4/cond/Switch_4 siamese_1/scala4/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_1/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/batchnorm/addAddsiamese_1/scala4/cond/Merge_1 siamese_1/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala4/batchnorm/RsqrtRsqrtsiamese_1/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/mulMul siamese_1/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_1Mulsiamese_1/scala4/Addsiamese_1/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/add_1Add siamese_1/scala4/batchnorm/mul_1siamese_1/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_1/scala4/ReluRelu siamese_1/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_1/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_1/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/splitSplit siamese_1/scala5/split/split_dimsiamese_1/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_1/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_1/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/split_1Split"siamese_1/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_1/scala5/Conv2D_1Conv2Dsiamese_1/scala5/split:1siamese_1/scala5/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese_1/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/concatConcatV2siamese_1/scala5/Conv2Dsiamese_1/scala5/Conv2D_1siamese_1/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_1/scala5/AddAddsiamese_1/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
n
Placeholder_4Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_5Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
n
Placeholder_6Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_7Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
O
is_training_2Const*
value	B
 Z *
dtype0
*
_output_shapes
: 
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:,*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesadjust/biasesadjust/weightssiamese/scala1/bn/betasiamese/scala1/bn/gammasiamese/scala1/bn/moving_mean!siamese/scala1/bn/moving_variancesiamese/scala1/conv/biasessiamese/scala1/conv/weights3siamese/scala1/siamese/scala1/bn/moving_mean/biased7siamese/scala1/siamese/scala1/bn/moving_mean/local_step7siamese/scala1/siamese/scala1/bn/moving_variance/biased;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsiamese/scala2/bn/betasiamese/scala2/bn/gammasiamese/scala2/bn/moving_mean!siamese/scala2/bn/moving_variancesiamese/scala2/conv/biasessiamese/scala2/conv/weights3siamese/scala2/siamese/scala2/bn/moving_mean/biased7siamese/scala2/siamese/scala2/bn/moving_mean/local_step7siamese/scala2/siamese/scala2/bn/moving_variance/biased;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsiamese/scala3/bn/betasiamese/scala3/bn/gammasiamese/scala3/bn/moving_mean!siamese/scala3/bn/moving_variancesiamese/scala3/conv/biasessiamese/scala3/conv/weights3siamese/scala3/siamese/scala3/bn/moving_mean/biased7siamese/scala3/siamese/scala3/bn/moving_mean/local_step7siamese/scala3/siamese/scala3/bn/moving_variance/biased;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsiamese/scala4/bn/betasiamese/scala4/bn/gammasiamese/scala4/bn/moving_mean!siamese/scala4/bn/moving_variancesiamese/scala4/conv/biasessiamese/scala4/conv/weights3siamese/scala4/siamese/scala4/bn/moving_mean/biased7siamese/scala4/siamese/scala4/bn/moving_mean/local_step7siamese/scala4/siamese/scala4/bn/moving_variance/biased;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsiamese/scala5/conv/biasessiamese/scala5/conv/weights*:
dtypes0
.2,
�
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_1/Const
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,
�
save_1/AssignAssignadjust/biasessave_1/RestoreV2* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_1Assignadjust/weightssave_1/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_2Assignsiamese/scala1/bn/betasave_1/RestoreV2:2*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta
�
save_1/Assign_3Assignsiamese/scala1/bn/gammasave_1/RestoreV2:3*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_4Assignsiamese/scala1/bn/moving_meansave_1/RestoreV2:4*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
save_1/Assign_5Assign!siamese/scala1/bn/moving_variancesave_1/RestoreV2:5*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save_1/Assign_6Assignsiamese/scala1/conv/biasessave_1/RestoreV2:6*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_7Assignsiamese/scala1/conv/weightssave_1/RestoreV2:7*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`
�
save_1/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave_1/RestoreV2:8*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave_1/RestoreV2:9*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave_1/RestoreV2:10*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave_1/RestoreV2:11*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_12Assignsiamese/scala2/bn/betasave_1/RestoreV2:12*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_13Assignsiamese/scala2/bn/gammasave_1/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_14Assignsiamese/scala2/bn/moving_meansave_1/RestoreV2:14*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_15Assign!siamese/scala2/bn/moving_variancesave_1/RestoreV2:15*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_16Assignsiamese/scala2/conv/biasessave_1/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_17Assignsiamese/scala2/conv/weightssave_1/RestoreV2:17*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
save_1/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave_1/RestoreV2:18*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave_1/RestoreV2:19*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
save_1/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave_1/RestoreV2:20*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave_1/RestoreV2:21*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
save_1/Assign_22Assignsiamese/scala3/bn/betasave_1/RestoreV2:22*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta
�
save_1/Assign_23Assignsiamese/scala3/bn/gammasave_1/RestoreV2:23**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_24Assignsiamese/scala3/bn/moving_meansave_1/RestoreV2:24*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_25Assign!siamese/scala3/bn/moving_variancesave_1/RestoreV2:25*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
save_1/Assign_26Assignsiamese/scala3/conv/biasessave_1/RestoreV2:26*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_27Assignsiamese/scala3/conv/weightssave_1/RestoreV2:27*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save_1/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave_1/RestoreV2:28*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave_1/RestoreV2:29*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave_1/RestoreV2:30*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave_1/RestoreV2:31*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_32Assignsiamese/scala4/bn/betasave_1/RestoreV2:32*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_33Assignsiamese/scala4/bn/gammasave_1/RestoreV2:33*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(
�
save_1/Assign_34Assignsiamese/scala4/bn/moving_meansave_1/RestoreV2:34*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_35Assign!siamese/scala4/bn/moving_variancesave_1/RestoreV2:35*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
save_1/Assign_36Assignsiamese/scala4/conv/biasessave_1/RestoreV2:36*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(
�
save_1/Assign_37Assignsiamese/scala4/conv/weightssave_1/RestoreV2:37*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save_1/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave_1/RestoreV2:38*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave_1/RestoreV2:39*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
save_1/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave_1/RestoreV2:40*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave_1/RestoreV2:41*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_42Assignsiamese/scala5/conv/biasessave_1/RestoreV2:42*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_43Assignsiamese/scala5/conv/weightssave_1/RestoreV2:43*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
�
siamese_2/scala1/Conv2DConv2DPlaceholder_4 siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:;;`*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_2/scala1/AddAddsiamese_2/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:;;`
�
/siamese_2/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala1/moments/meanMeansiamese_2/scala1/Add/siamese_2/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese_2/scala1/moments/StopGradientStopGradientsiamese_2/scala1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese_2/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala1/Add%siamese_2/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
�
3siamese_2/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala1/moments/varianceMean*siamese_2/scala1/moments/SquaredDifference3siamese_2/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese_2/scala1/moments/SqueezeSqueezesiamese_2/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_2/scala1/moments/Squeeze_1Squeeze!siamese_2/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
&siamese_2/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_2/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_2/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Nsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_2/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_2/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_2/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
�
Tsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_2/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_2/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
g
siamese_2/scala1/cond/SwitchSwitchis_training_2is_training_2*
T0
*
_output_shapes
: : 
k
siamese_2/scala1/cond/switch_tIdentitysiamese_2/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala1/cond/switch_fIdentitysiamese_2/scala1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese_2/scala1/cond/pred_idIdentityis_training_2*
T0
*
_output_shapes
: 
�
siamese_2/scala1/cond/Switch_1Switch siamese_2/scala1/moments/Squeezesiamese_2/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_2Switch"siamese_2/scala1/moments/Squeeze_1siamese_2/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_2/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/MergeMergesiamese_2/scala1/cond/Switch_3 siamese_2/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_2/scala1/cond/Merge_1Mergesiamese_2/scala1/cond/Switch_4 siamese_2/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_2/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala1/batchnorm/addAddsiamese_2/scala1/cond/Merge_1 siamese_2/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_2/scala1/batchnorm/RsqrtRsqrtsiamese_2/scala1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese_2/scala1/batchnorm/mulMul siamese_2/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese_2/scala1/batchnorm/mul_1Mulsiamese_2/scala1/Addsiamese_2/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
 siamese_2/scala1/batchnorm/mul_2Mulsiamese_2/scala1/cond/Mergesiamese_2/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_2/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_2/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_2/scala1/batchnorm/add_1Add siamese_2/scala1/batchnorm/mul_1siamese_2/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
p
siamese_2/scala1/ReluRelu siamese_2/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese_2/scala1/poll/MaxPoolMaxPoolsiamese_2/scala1/Relu*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
X
siamese_2/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/splitSplit siamese_2/scala2/split/split_dimsiamese_2/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
Z
siamese_2/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/split_1Split"siamese_2/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese_2/scala2/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese_2/scala2/AddAddsiamese_2/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_2/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala2/moments/meanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala2/moments/StopGradientStopGradientsiamese_2/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_2/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala2/Add%siamese_2/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_2/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala2/moments/varianceMean*siamese_2/scala2/moments/SquaredDifference3siamese_2/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_2/scala2/moments/SqueezeSqueezesiamese_2/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_2/scala2/moments/Squeeze_1Squeeze!siamese_2/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_2/scala2/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_2/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_2/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_2/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese_2/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
(siamese_2/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_2/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Isiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese_2/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
g
siamese_2/scala2/cond/SwitchSwitchis_training_2is_training_2*
T0
*
_output_shapes
: : 
k
siamese_2/scala2/cond/switch_tIdentitysiamese_2/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala2/cond/switch_fIdentitysiamese_2/scala2/cond/Switch*
_output_shapes
: *
T0

Y
siamese_2/scala2/cond/pred_idIdentityis_training_2*
_output_shapes
: *
T0

�
siamese_2/scala2/cond/Switch_1Switch siamese_2/scala2/moments/Squeezesiamese_2/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_2Switch"siamese_2/scala2/moments/Squeeze_1siamese_2/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_2/scala2/cond/Merge_1Mergesiamese_2/scala2/cond/Switch_4 siamese_2/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_2/scala2/batchnorm/addAddsiamese_2/scala2/cond/Merge_1 siamese_2/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala2/batchnorm/RsqrtRsqrtsiamese_2/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala2/batchnorm/mulMul siamese_2/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_2/scala2/batchnorm/mul_1Mulsiamese_2/scala2/Addsiamese_2/scala2/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_2/scala2/batchnorm/mul_2Mulsiamese_2/scala2/cond/Mergesiamese_2/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_2/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala2/batchnorm/add_1Add siamese_2/scala2/batchnorm/mul_1siamese_2/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_2/scala2/ReluRelu siamese_2/scala2/batchnorm/add_1*
T0*'
_output_shapes
:�
�
siamese_2/scala2/poll/MaxPoolMaxPoolsiamese_2/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
siamese_2/scala3/Conv2DConv2Dsiamese_2/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:

�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
�
/siamese_2/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala3/moments/meanMeansiamese_2/scala3/Add/siamese_2/scala3/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
3siamese_2/scala3/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_2/scala3/moments/varianceMean*siamese_2/scala3/moments/SquaredDifference3siamese_2/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_2/scala3/moments/SqueezeSqueezesiamese_2/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_2/scala3/moments/Squeeze_1Squeeze!siamese_2/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_2/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_2/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_2/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_2/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese_2/scala3/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_2/scala3/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_2/scala3/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
usiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_2/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_2/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
g
siamese_2/scala3/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

k
siamese_2/scala3/cond/switch_tIdentitysiamese_2/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala3/cond/switch_fIdentitysiamese_2/scala3/cond/Switch*
_output_shapes
: *
T0

Y
siamese_2/scala3/cond/pred_idIdentityis_training_2*
_output_shapes
: *
T0

�
siamese_2/scala3/cond/Switch_1Switch siamese_2/scala3/moments/Squeezesiamese_2/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_2Switch"siamese_2/scala3/moments/Squeeze_1siamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala3/moments/Squeeze_1
�
siamese_2/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese_2/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/MergeMergesiamese_2/scala3/cond/Switch_3 siamese_2/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_2/scala3/cond/Merge_1Mergesiamese_2/scala3/cond/Switch_4 siamese_2/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_2/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala3/batchnorm/addAddsiamese_2/scala3/cond/Merge_1 siamese_2/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala3/batchnorm/RsqrtRsqrtsiamese_2/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_2/scala3/batchnorm/mulMul siamese_2/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_1Mulsiamese_2/scala3/Addsiamese_2/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
�
 siamese_2/scala3/batchnorm/mul_2Mulsiamese_2/scala3/cond/Mergesiamese_2/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_2/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala3/batchnorm/add_1Add siamese_2/scala3/batchnorm/mul_1siamese_2/scala3/batchnorm/sub*
T0*'
_output_shapes
:

�
q
siamese_2/scala3/ReluRelu siamese_2/scala3/batchnorm/add_1*
T0*'
_output_shapes
:

�
X
siamese_2/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/splitSplit siamese_2/scala4/split/split_dimsiamese_2/scala3/Relu*
T0*:
_output_shapes(
&:

�:

�*
	num_split
Z
siamese_2/scala4/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
d
"siamese_2/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/split_1Split"siamese_2/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_2/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_2/scala4/AddAddsiamese_2/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_2/scala4/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_2/scala4/moments/meanMeansiamese_2/scala4/Add/siamese_2/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala4/moments/StopGradientStopGradientsiamese_2/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_2/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_2/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala4/moments/varianceMean*siamese_2/scala4/moments/SquaredDifference3siamese_2/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_2/scala4/moments/SqueezeSqueezesiamese_2/scala4/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_2/scala4/moments/Squeeze_1Squeeze!siamese_2/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_2/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_2/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
 siamese_2/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_2/scala4/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_2/scala4/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_2/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
"siamese_2/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
g
siamese_2/scala4/cond/SwitchSwitchis_training_2is_training_2*
T0
*
_output_shapes
: : 
k
siamese_2/scala4/cond/switch_tIdentitysiamese_2/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala4/cond/switch_fIdentitysiamese_2/scala4/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese_2/scala4/cond/pred_idIdentityis_training_2*
T0
*
_output_shapes
: 
�
siamese_2/scala4/cond/Switch_1Switch siamese_2/scala4/moments/Squeezesiamese_2/scala4/cond/pred_id*3
_class)
'%loc:@siamese_2/scala4/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_2/scala4/cond/Switch_2Switch"siamese_2/scala4/moments/Squeeze_1siamese_2/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
siamese_2/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese_2/scala4/cond/MergeMergesiamese_2/scala4/cond/Switch_3 siamese_2/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_2/scala4/cond/Merge_1Mergesiamese_2/scala4/cond/Switch_4 siamese_2/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/batchnorm/addAddsiamese_2/scala4/cond/Merge_1 siamese_2/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_2/scala4/batchnorm/RsqrtRsqrtsiamese_2/scala4/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_2/scala4/batchnorm/mulMul siamese_2/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_2/scala4/batchnorm/mul_1Mulsiamese_2/scala4/Addsiamese_2/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_2/scala4/batchnorm/mul_2Mulsiamese_2/scala4/cond/Mergesiamese_2/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_2/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_2/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala4/batchnorm/add_1Add siamese_2/scala4/batchnorm/mul_1siamese_2/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_2/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/splitSplit siamese_2/scala5/split/split_dimsiamese_2/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_2/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/split_1Split"siamese_2/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_2/scala5/Conv2DConv2Dsiamese_2/scala5/splitsiamese_2/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese_2/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/concatConcatV2siamese_2/scala5/Conv2Dsiamese_2/scala5/Conv2D_1siamese_2/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
�
ConstConst*'
_output_shapes
:�*��
value��B���"��-�=4��<�k =f��� _�;��U�,_�=��ټܳc=!<��6;��C��(�*�z��%��V���
�X�K=p-<���H�(��=x$�$ڕ<����N��h�= ���V� ��;\d=п���ü���<Rd=��<ܑ�<�S=�*�=�Y���=�d�� y�;ˎ=�M��F�;���<�+<�U[���;���<N�&=4���sv�=��<̛�<�9;=�nM;��=2Ň�%�<=X
e=�S=�b=���H��Ƞ��p��t��]����m�:��;�S��ʉ=�`E=������T�?=`�&;��o�s`���=3= :#< �7:��1��t=�ݶ�t�O��м�N=�h�T��<���=�J�1ߐ��|�$<��`�;�Or��ꕽh1M��һ
�V���t=�J=���:�����T=�'�<w�� �=�9ļK��=���<.s=���϶�=���;�֟<{��=�&�兼��=�Kz��=4m��d5=�=.;��%� 7���m]���R��$c<p��l��<@�"���]�0�3�0���з����X���"�8��< ��9���� 4�� �~;@��: ��� $����߼@�:�=ȓ�����LK���x��d�����<(���w����� �����=�
8� ����V��01<��5�R�N�h*2� �g�h=��w!���q��3��@�;:�<<ߐ<<G��U=Z�w���=�Ug�x@�<�:�:�1Ҽ��������;8I�(��<��&=t�������s���c��TT�`�����<��A�L��< A���r�p,x<��X�Z��`= ��n%��ű�D��d>μ�a�.���9\�����X���@��� ��M<0��;�mn�@���v����)�l���I����;x�<H�z���� �ǻ`�����e= :]���F=(J���ջ@[� �`�Pʄ;�0Q��<tb���@az�@���z�; ���@�>��"=@ڴ�g�n=X��<���<�@����9;�tֽ ��<0�'<� l=8c =�*�����P�K����,��.�,�μD_�<��=����y�@θ�%�=P���c�%=��Q�Pe3=p�A=�<r� �$g�<�Y�<PW�<�d�� �r<�fʺțM<��={�Z=��=$�<�V~=���<���<�E�����<�nJ<�+�=4�P=�˄=�!:=�m�<C4N=�=�Gp<��='A0=4��=�|��5�Ư =���<�֥=����� Bu<<��<�X��9��ʶ<�l���<��M=(�<Xfk<f�x�4"2=h:p< �d���� Ѽ �ڻP�v<x�v<�u�<��u������-�X\l<��ѽ#;�=�]3=L��p�H�r�=�����"R=���<�{�(x����<�]ż�������<�=x���%�<v��=�'{���I=��=���.�E=�E�=L=P�ệ�x=�9����V=���=L�<����(�)<��9���`=0O�of�=K�H=@d׼��B�H�R����? =�PW�n/
=���;:q��;�,%��H�'�~�w���x�u�X�=��<��C�v�/��jɻ��<�q1=ļ��2�� �G��!=<�#� ��;R�7�^L0��u漧2d=|�'���i�ЭR� /��ѼG����q�l˼$�#����u	��p��;��^����r���������)�: y<d�<)��T�=�0���:=n�J�/==���<B�Q��b�`Z����O<�fP<P=l�h=D$����2<hv<T_f��E�;XcF���<����;p9������<��i��k.: 8�<�m�@�<� �y��d����Ҽ@!�;!/�P��<�O<�����H=<� ��K�;p�< -��v�=����`��+<��k<,���I�<V���h��`j�;�j��ܩ=P׵��<���@��<𳋼��?��j=$4Q� 7z���;(+����4n�<�ܥ<�� Y�XRc=��)�͊i=�W8<�*X�xP���j����E�'=�c=D#t= �:r)/����� ��<@q<u)���v���=���<'I���#����S�b= �ݼ䮑=́���=����ȇ<��
�8�L����<ʝ�=��b� =���P����H=�L=���=���="�=h9��Ј<�񭺐Ď�p�̼ i����S
>�)�=eC�=�z2=��=@�:#h)= �:�傻��=�o�=��5�l��� 6B;�UC�r��=>g)��o	��k6=̐=
�<ؼ��:=wؚ���w= )�; ����;=:v0��+\=�h<�'�=_v�_�������7�<�,=,+�(�q�U�<Ċ\�06A�� ���=0�e��f��G;'=Y�=���<ׯ�=��n=��Y�����*�=�J���tպ�n�;�"�<�|C���=fc=��R��+�=82]=L��x�Gh�=��߼��; �B<n�I��S�=)uL=:�V=@�2� �%��i<X�<���t>o5�=dB��.�� 3M��qr��o=lO�R�=�H�<��n����0?׼� 9��%���=����<���=�q"=��"����� _;�6e=�+�=J����̽��h<��U=��/��=o@���
�x�b�=<Y���A�8Zr�@����.�����<@ؼ�������J.�mA�B/=�_#���w�8�G�?�<��X� �y�(I<�U�<��l��>�5��=�K����y=�=�oe�  _�l6Ӽ=�M=7�\=`��<�6i=�k¼�DY=��d=�A����.=��W�h�=Jsc�x 
<�
� 
V�l��<ϣ<;�=P��<�%`�p>�;��7=�Cr;�-���p=��D�G��=�Zh<�Wڻ���<����T`�<�!=Tؕ<X��=��S�"9=��q�6n=f�5� �6��/��� ��<�Ἦ�=���� `F�l����2=��ü`λ�"Z=R ���
�hm�=�o=� =/�1=ҙU=P�`���r<�M�=���� �=���d?������dvK�ơ������=���<{��=����^�~8o�{n&=P�+=��彐u�;$C�<`��C� ��;>!��4<L�2�vE�=�.��ft�=
��� l<��ҼJ`:���g<7��=H�O�n==²1�����J�<�=��=��>i��=�:��8�n<�����	��dⰼ����(@���>�	�=���=��=Q��=Ԕ��U<<ඊ��jQ��p�=�3=�T�;�V��()��D6��[��=fQ[�TxJ��@'=��=�l=�?����(=�ʽF�E=c�\tb�P�=�e��&^=�;ә�=�����|GI��F�<��<~���1W< �J=���L(���Z@� ��=:jM�|�����=*֕=�u%=���=%[k=�M�,�*�S�=��P�0!�<�h��G#<@ST�>뻕�A=Ǹɽ���=G�=��*d�"
>V+��?Ҽռ��Ž�q=H��wFE=l��0Z��>�=Э�������>NX�=X�K��:���]��H��S�h=�$��<l��<�]�7�׽t�����dĽ0�R���=��=�=p����������<�A�=���=t׼�dϽ ��<{�V=�h�i�b=ij��q5���(�X��=���������B���������3K���`1��b0��a۽��;�н�r=h���&� �q;T�<Vs��;���\�;��I<Gӊ�tF">@��2�<k@ս�\=�=��<��[�,"��x�=���=�l<��=0|d��6�=(�=넽�|=O=�?�<�V�����< �	��ns��D�<��=���=���<�i���E�<6'�=���<`�����3=�WR�h/�=���<�$]<���< ���t��<��6=?�=m��=������K=��S��6^=~�1�0X��X0��0K<8 �<�3�糓=�̧�x�r�\����g=�����K9�rs]=�h��f�3� >�Z�=[L=yoD=P&�=Р���=�ё=�~���K�=�]K����G���Ϙ�*W������]=��;�e�=@ވ����K����=��C=���x��<P͆<��ӻc!̽|]�<_�x����8�,ҭ=Rͷ��`�=�����=�;��]��f�����<��=$�h�GV=�O^�H��@x�;��,=���=J��=[U�=���� �9�����@��b!��}�fY=�E�=���=��<��>�T����D��hI���gf=X��<x�&�� y� =�����
�<��\�|�C��=/-�=h}�<�f���Q�<��Ž0��;�g�������;_����%=x�x��ɴ=��;��ٽpD��4�; ��:��̽���:X�M=N�ͽj�V�`3�kE\=�$� �[�R�=P+=Ⱦ=DQ�=ɹ= �����r�s��=���M�;=G����J��|���&���<!���=���<h�I�����d�>�d��׻��^�%�����)I=�qj�< =5߀�����Q�;=�V� �8�mQ�=�L�=��$��P��P���H�<#=$��� �-� �<����i��<�ּ`�R�Gڔ�@�����=�|�=�z{< ���j��+y<�N}=]ߥ=��;�|�H�><�O�<Ȥ���&2=�gL���|���w�=`�����(���?���3��3������ ;�P���1�M����N]=�=�����J	<��<z����@�h�m< }<��X����=��8<,>�<g�����=�g�<�:����;�c�6�h=�K=�Z< �<Pn�;ie=�$=�E���=�m��i�<�s/�`~�<X旼X���"<<8�<�Ka=��N<���@�;
�=���< �=�@�<4���C�==\-�<�z�<��x<�/}:P�;iO6=���<�x�=P�7<�~=��m���L=�V	���3��ӻ0Ԃ;@H�;`���/" =(�'��.�@A��dnF=�̫;�|7;f[P=��l;�{��<�=�9p=��'=@��<��"= ��8�=�[3=�te�;��=�����E�D�x۽�i�0�� ��<��<{�y=��U����Ľ }�<�=�������<�9&<P��;�>����<l�輨�{��$��A=tv����=v����<1�ϼ���XY=*©=�B)�7��=5F)=<���8�����_<�؎=�V�=B�'=8�ۼ�K�� �m�
ڼ��C;�[6�$n���¼";=/��=�vV�LM�=^ڜ�.꼔��>���7=���<��m�(�T��������ବ�b ;����8j=⧛=W�<hV���?G<�ۗ���P���@�R;�ļ�oW����<��D�z�[=|�<��|�X�}� �2��]���j�l���@�+<2���������xN�<h$��h�1<p黻@򁼈�C<g#=ȸ7<>�<L�����=��;���= C���[�謧�`�����ݻ��˽�.<=�e���B���j�?��=8\��5���~}������=�H�����<oK��fF����3=����X3<}�T=�<�5,���$�	��/3�d9�<�N�0�L��Ŷ;���IO���Ҽ(�3��ta�b�6�s=-�P=� ��@C���愽��8�<i�<tj=p5�;���������'�з���i<��%�_Y�W���3= ��9dH���3C��ڥ:H"�&��L�ԼQ)<�b�<��6.�˕"=p�l�@����6��L=�$㞼X�N��� <PN�;�[�r~�=Ծ�<���<�۱��/<8�f<6�t�h��<�	����<�ǆ<�r�@`�`r�;w#= �~; ���`Y[;�ץ���=.�� 5ں�Z�P0�� ^��0�Ȼ%zQ= ��9l9@�*̻Z9Q=��U<�0׻����n���	�<�$<<���<�I�: �չ47���6W<:B<�j�=0w-<@�6;@g�;	=��������绀�0�d޼��/����<���@�O��k;n�=H�<��ܼhS=0�<�0RV=�t6=�<h��σ;������J=��j<�������<�a<q}��諾$SR� b��Z��=*<��r��=|i=��	�f=5�@�S��V��O���z��۫<��t=	��= �:�zt��擼���<�r���Y= 5}���*=Pzu<�H�<�3�D5�Vp�=p�a���,�#a=nŒ=��A=V=�2�D��=�3���= ?���;g�;=������ܼ4��<����Z��l�=�o�= Ka�6�X��<��<p�N<�k�<�M�<��>��QM)=�&=�哼�-�=d,�� �ƺ��X<<L�<�P�<�pG�4�����μ ��:i�=�\�=��������=4'���$��#/=�<�F<4�ἈH���νp�p��c �@ ���K��r�<�XW=�f8������X��T8�h��<؃{�(���ʽ觞<�,B�XL$<pc�=��;4F����=$�<��"<`�=<u�<(JW��g�=��{=��;p�Լ(�M<Ф�;��=Ӏ4=̍C=�99��wX=�7A���<,\���N=�e$���ۻ O�;�Xz�^3��s"=@Vλh�Z<�����f?�f+�07� d�;2�5�(�ɼ�w���<�#ȼ���:������<и<���<�Mż� ��psP��'=���\�ؼj��0&���曼^D"=`��`*ü��C�0K�;�C�:��ټhI<��h���V���,r< ����P��J�W� %���.�в��0�;,#<.��x��=G�Β=r����<`�]<QB�\ꤼ@��PR�<@k; �M<���<d�뼈{�<`v�<������@nM�t��<�K���z�@�@*�:��<󃻤�<h��<�����< B��L��� f�������9G� �;���� I�;���:<9�왃<�p���y:�S�<Bu��<�ۼd۳� �.����:�jO:x<����b;p�0l���.=�f����2=`Լ��<�r`�J
����ż�����f�;�䡺�ٺ���<x\<�����S=V2=��g��.&=��;�wͼ��w��|�h�_�\ �<|�C�lQ�=�l=�2��������D< �+����� �<wĘ=��3=�X�8\���7s��9=�Q���l�=��Q���I=XaC<�a�<��!�`�;���=[�<0���HE�<$�<O=��=�Uq�e��= g��S�=8����ջ�!<f�����2�<��弘<�$�=��=�[ʻ��D���<|��<��;�<�"Ї=Vn>rS�����`�M<���3v�=P�Ѽ Q�:�g3<bT#=��=V������k��Is=`�<˪=�����Nc�ȉ=���� ��9F\�̗��؅p<l�<��^:�q����D%ּ��޼(;��餽K�?=(�<<�-» M���8����}=���<�������{5=^)D��@<�+?=�!=�ߔ��\�=�|�< �]��!l=��5=Ly^�A�O=��=0�8�|��<���<�H���-a=��-=�0�=,+��`�}; �&; W�;|g�N؀=���<|¼ ��P���U�1~v=��O��N]< ]a;־A�N�t�$3�� ��;���� �@��:(|�=��|�����`���A< HT<]=v��r 5�p}e��%=`u�� c4�^.�x�-�X�ۼ1�Y=�����*��E��XU#<؊��RdS�Ğ�`��;H�3�T���nhR��׼<�(λ����U��|���&�H<*� Ψ���;b�D���=8���DՏ<y����]I=X-=����p &��	l��̵<���< ��;h��<���AI*=y�=t�X�@�!; D�h�<����H~��(���±�'	=@�g�R�=�6?<L���|ƻ<`��;X�W���;xs�~w�xʷ<�΁�`D4;�)<,�ü@��;@��: �]:0�;=LU� "л�<��x�8<,㹼 q,:��&�1F<`��Ȑ���qE=��(��="����=�..�����><�<����̼�	�<�$�;��D;
=$)�<�^��Sh=��4= �ߺ�{'=�<���3�(�]J��_��l�6�H-l�=�L=�:���*�m�����< R�<�ݽr+&=��=���<���U������z�<�ʟ��S�=a��~�s=����߃:J�4��4��"�L=r}=8F�xo8=�Bٺn�=;�2=�?����=��<>�=<�
��6�������\�v$N��"<d�I�C�=g��=���=��λ W�<����H��<@م:��;��b�=���=�i?���t�p܋;D������=ߢ��Hݻ�V�<o�=�ej=��ڼ��;�o��K�= /;x� <�к��S�"�=����.�=v^�E`��PS5<�"d< ?C;�mJ�\
�L������z���ǽ�̈=M˼|����;=�9}<���s1�=�m=��� ȗ��h�=�]?�\��<p��<�2=8=¼((�=̷�<rWE����=,�=�M��������=4u8�p<1=@.�"�>��5c=�ξ<���=�����*��,�<�Ĕ�Ƚ�4��=p@T=8�ͼ�oG���D�����c�=t>���}<��b<���-����)ż�$ <%����:&�,I�<\�=p�������a8���/�<P��<^��=�&%�j�}�P����=�f;@O.<�?���Ѻ������=��إ)��O��@�D;8�����n�VG��ş:����ƚ���o���=��� wc�f� � X�8:z@�<⑼�����&�.�S�A> �k�@S�;=?���u=)�=��o���;�PT�C�0=<�!= �ڻ��<l�켍s|=N�c=�v��|�<��鼈-X<�����	�����x-W���=�k�; -e=�k<���<'=�/ =�HK;ؗ<�A
;jso���=�k-�`R&<�n�<��h<h<XQ<���=���h�F<���0��<����8�^�0(\�p<�<�(;�
���H=�4��˔<DM���F=�$��R��͜<�˹����2�=p��<蓈<�%0=��=t,ż�}=�G:=��M�W�Z=@I�6����D�f�J˫�������< N;\�=,z��%U���Ȁ�"�	=Ц�<���
=^�=�<�(ܽP�X���)����:(yż��=(@���ч=����Ǽ*W��I� 3S<(�=�Y�Uf`=��j� ��;���<Pis<㭄=�J\=��=H�d�Xu
�o�� ^��;0��Ő�\4�����<!��=��=���:f�=@=��䍃�`?��x���=��,=0�� ����4���Gu���=�,�������<s��=r�d=�:���D��޽�&�=t?��Hꄼ������w�1=�5����<=r�:�g�Խ��v���;�輼�V��U��P�S�>%��@T������uT==;伌d��_�=�F�<�"_<��_=��R=���v兽��=�ؘ����<p��;<��<��ռ��j;�=����@�=t.=�]�W����=�N�P)�<F�S���=L
�Ds�=��o�f�WmN=Їμ������=�|=앮��8I���o�-V��,�w=4�ټ�7��2�<�����ȽPj �H,<y=��8�2���U=�.[=�������;e��@�
=�$=/6�=���z�w���޺���<�!<�g�<:X����<���U��=Xd�x���T���B�X�� a��>�@��;ܲ��^� ��vE��&3=��8�Мﻌ���P,�;^�[��}�ݼH���Mg��s>�!�<�ሻ�Yҽ0]`=(��<�>6����;�����v=S�L=�v� �;�p��nę=�x�=�A]��i =dxϼ kG;'k��x
��Xe���햼�^�<���<�{�=P�;��1�5f0=�>=�2�<P�Q<`iT;�ŀ��= _���:�<TK�<����3<xv;<�Q�<K��=���l��<���)=���8v�� �@�<��< ��;d�V����<�����ȫ;x/Z<��b= 
ݻt2���N<@w$<j[�E�=
i=tL�< ^$= ;=�U��X�={)=�/�g�=��̼��ؒ{�r���Dh�����%=����w�<Ŀ鼕���5)Ͻ���<@5<Hh�� = ��<�<�:3���0i�lQ��>��T��e�=h��0�=����%:���������`x�;���=�����M= Lٹ8�y�@��:BO	=�JT=�vd=���=v�����R�50���H���޼���͹��g��~��=)��=�p�:���=��9�hZ���o�,����Y�=�q5�8}\�H���Dӊ��w��R1=8���&k�P#�<t�=�{4=���P��u���Z=`j���ƚ�0᧼I����=��(��o:=�=ּ�d��0��@�A��`��[��E׻���p�� /��t�O< �=��_:1�t= K	;0�{<��<J]=H7<X������=�����>=�Π����H��X�9���<k�۽�l==��\�=뛽���=�6�#��Z�>�����p�<�y���c=xؽ"���#ؔ=4�E�PҜ���=�>
=��ȼl���:R��ۋ��8#= '����ڼ��<�������r߼ �%<�?������[=4=����%�;J�a��b�<�=&��=h�����H@\���<�S�<HN�<��"�p��;ȁ���4=�zn��?ؼ�t�P������r�(��&����;��;��$����[�=�c �@���μ@@��>J9����L򋼈嗼��;�v�=�f=��:Ay���
/= i�<��;�@��;�0\� M"=��=nf&��a��R5�GkV=��=d�?p<�Ԣ���������0~޼L����i*� ��9��+i=������� �=@J�<�׍<��)<����q���o<���w�< V-9��:(�h_m<hE<!�=�o9<�Ė<�_�|b=���˼�6�`v�<��S���I� �; J�@"�; �<C=��;��8��_.<(�b<�ϼ�A�=��6=��<ч<�_�<���"Tv=�ˮ<pM���}=�ˊ�;��<ӽ ���P���x�2=��^< �=j����ݽ�<��L,�<@�T�d��<�Ã< %�;��H� �<��q��=T��΋�/,=�� ���=�p�����k��F����<�=�I���==DD�<�����}���Ŝ<[�U=���=,Ԓ=�۽ �c�H�	�"F-��$ �da��cֽ�\l�/+�=0*�=������=�2��T^���� ��*c<=�¼��B���.�����!�@Ao;�L����=ܼ�=Y25=��*��-���཰�<`"��������A%��l*�<\v��j�,=@#��`̽ �A�����\x�~&���������2�H���佈�D� ��;8��<`��;����c�:��{; �|<Ж=����w��=}���"eG=�f���t����t�>���<�M�V�'= CB����;���8�=�C�1H��ύ����.C��W� ��<L���:��l�=�M�������
=��(�R���P��d@�j|�4ӵ<�f����ϼ�҅;�F���q�l�� ��:���n�x\=�<>Qa��v��J�H� 񆻨"d<,VN=P������h��h���P��<@T<�f� ��9���p��< �;���8샼��a;����4	�B�����;�#����D��B��Z=�^�`�;���x�n�$B���9;�`���TƗ�T�Aa=��= =/�9����=�s�<��c��g$<�N�`�p< ��;Np�p/��&=��!= ��9�ڼ�@�5��z����ݻb�h�:G�����7�H�V�P���H7\=�5�����dc�<�-�<з�;�!;�F$�s2�����;0�ϼ 6y<�����;�f3�`>�� �ͷ/��=�y�<p5��_<p��<R�H���������F<z$�` � '>� {;�1�l��<��.=8L�<l�0�	<�t�<���GP]=�|= �-��]'�`�(�����2�=�*:�� ����h�ռ����,c�dw{��J=���=�z��%��=�T�=0ƻ�N���r�<��;�F�S���v�:=���=È�=�<e<�#�t$���?��0"&��e=�D�;[�"=�"�JJ-=���:��=4JJ��U;=�nZ<�>=�Y�=��=�ȶ��y!=V���fz<0y�;�2���=v_���K��G=�Ѽ�:ƽW!�=�ҝ=�3�;Խ�V��tHQ=�*j� 	�:���<Vk">�	���{<���<�㕽}5�=�Rl�`��;x�<�P<���=�B�����[���i=@l<gV�=h6��F�m����;ĊJ�d�ռ�O�*��=O�<#5S=��;�"������B�@�<,��/����u��=<��h;�t@�7/��1-�� *�<�#��@1E�ڿ���=QЁ�8�<�
�=2k� *w9�*�=4��<Զ=����<8���r��d��=�.A=�K����7<�I��Jc=��<rO=��i=F����==��u���$; ��Ȁ�<�r� !�9�{�<��B�>�M�r<X=�񆺠";�N�����ZO��[-���<0����P���%,<&�@���<6�.�>{=P1���=����,^�<U��t= ߾<���0U�����<�|��} =��k��q;P?�; �<�^R<(FW����Q\=Pe�� �0��A<x�=�D< <�9N�e� �ɻzm�D����>���+�������= ��< �{<*>1��A�<@�v<��=����;+.<؜=��<�}ȼp��B~#���F=*�=�%	���܏�<�%�<!Ӕ��������;$�<�=����y�=�������;A�i=@�_< Ղ9_62=T毼b�s��ގ;ĥ���r�< ��:L�ܼ�$�;����3g<��	=Ps�t����t��fC<|M�<������ռ��=�_�p�ƼX_<���(�.= ��<��< W�:^�Z��9����<����̧�<��<.<<�!$=4��<x�Y�>��=�h�<P8���Ƽt//��X���0P�`��n=b����%=�#�=0�ӻ�54���;�n�< u�9?��N�@=��=k�C=�_��54�`z��0R<�ʼX��=��'��J<�����=.���mT=R�=8ӛ�F�= *�;�ߑ�J/�=���<�yc�`M5<�(�����< �����@����q���!P�,T=�@ż0�߼�>M�q=5��Ƒb�pj<@=����ȵ��@P=iz�=:����t�����:����v>6������ �6�@̯�E��=���������#��=�k<b�L=8l,��6!���;Xt|����#o���r<h�v<s==H�!<Y�N��p&��0ʼت#<n(]���<��̼��L;���<ZM9�b�D�r_:=lt�<�p���l��x�=�都XL-<�(2=��=ؿt<:�=���<���<h"<L��<����E+=| =�-W���=`�;�/<(�V<̃�<-o{=��]O<�维�<�9м@˦:(Em�X>��P<dlʼ��a�{_= E�;P��;@�ͺa��8�������5� $\�T��@�*�K�=�(��c�<���Ѯ<��M�P��<�y��� T<H���$O�<�A=|��K��ڄ<�a���Z�<P;Ȼ�W��h�<���<Ŗ<H����J���=$����x���܅<4N�<��6< x}�b)N��\廾�%�<K��B��U��� � ��=R=��(;��)��\ =�x�<��6���<�<hN�<��<����8x��,�hY=�<����p`��$@�<Ҏ<�m��Ғ4���:<�ˤ<��=�߻p_= ����-�<3K8=�Iz<��u��cp=�!�r�R��u�;>��p�<0>�0����:�h���z><DT=��� �9�{�;��)<��< ��9l�	�� �<x�"�������;`���m�[=���< f
=��5<�F�(&���]8=�&��l�=p)<0=�;4I�<���<X��[Q�=��i<�h�;0�Y�`�I�d.弼����"%��A����<�Z;�F���v=g�������Ǽt �<`�!ĉ�H)D=�=��<ۘ��P�^�0��;�^<0O��!ͭ=LE������D�; ������Ȟb=�.{<P�:< $0;���<�G�hHh=��;pŽ��»�?��=p��3S���6vR���e�?	=��� �q��p�=�=����0℻���<|N�<�#���0�V�a=[+Q=�����F��$S��1<���=�;]������=��b/�PY=��Hб�1��{�= ��;x��<X��C�� <p��;�a�EG��t��`��;�h�<�*��d�<�
��C_��f,�a�B=/d����<����KĻp�=�4��xe���� =��=}�����4��<��W����;�Ȅ<�"�=L��<R<=\�<�|��8<�$=H��|#����><�zS����= QA���"���� X�9[D=pż�@��j<Py�<t��@�<��m��h�`�<x����m�_c?=PN�;P��;���� ���� �����: 0q��7�3<|��<�90�'
=�Լ�<@»\)�<@_d�`��;PVN�|��<��#=ب0��d����<@�x�<�x�@���0�'<4��<���<�}��=�S=c��x��<X��<H5#<�����@z���*������мЌ��D,ռ�b�=Yh9=��b�/�4!=(x<�f����<�'�<�Y=|x�<�O���y��|��*+a=���< p��PG<t�<�� <����RP"��rb<�K<�ư<@� ;�=u=@9���V[<g�7=�m�<`��;Tjo=(��(��'K;��`�<��9� ���i��h�L�$.�<�No=`e@; �< �~:'J<P=<ЙĻ@$�&�	=���Ȃo�P��P�»��c=5V=� =0݄<���ͼ�lU=H͐�W4Z=��<��;4��<��< �h*�=0�;@W?;���:\rG�����@�����;@����m����9:8��`y?�d���,�ܼ_6���ި<���旽_|=0;A<�R7������3e��|�; ĝ;4ɼ?�~=,ր�*8$�hsG<��B��������<؃#�l+�<���P`�<򼐸�<8ԥ��a<�Z���L����T=Po��������ʚp��-�@��:@#���$x���=��=�H��UK=x�<`-� ����J���b(=N�,��� <�x�0�ٻ��x=�̚=\>����u�p������(��<�S����[I��ȹ= DJ9 &ɺX
������Y<�M<��3����,���-���H�:������=�&_�4��x�^�?j=����X W�Ğ�����c:�=(���<�s�<L;�< ��ı8���{<�,���t;���2K=�F�<�i'�0��<R��L�:�t0=@�!��X{��o�l�1��6�=4P���{��F�t�\��V�<��x�����Ie=(�1<�^M���<8�D��17���#<��P�1���=�L���%�����k��zBh�JG��N�: Q�;@]$�,��<�)��T��jT=t젼B�==��<���<�7�� p�8 ~!�P�o<�=�~�;�Q��%6=�,L�`�<^"<���:��; Ay�,��<87��R�,����<P�%�0��J�=�\�<�ي;\򍼞R
� �����;��A������
�B�����=�8W=�мB�v�Ф�<�W��`RŻ���<t��<}[F=,1�<�霽�v� �i�ٟq=��<h�4����<4�< �/��������u<@�� t�9�p�;a%�=��Z;���9�X=@�<�.E<�%R=Vm��>�pn��27��ٻ<$���w���K~�(r���l�<�=�j�<h��<0'���a<�<H���d�8�= �%�(��`Y��D����o.=5Ė=ln=�-^<���� ���O]=�O�f�=�+=�G;xCH<���<`'_�W:�=���;�^��ȋS<d�K�Ȅ�<@=�; )*�4��<�gO���=*#��w�t0����o�����7�<����I���y�<@{b���Y������G� �� �_�l���/=42��ʼ@���wi��`�M�D뼔^k�z�=�n���<�D��'r��Hy�<Dwʼ�`��r��=.������Pr��:�t�������l���n=�Lc�<��{=��n� �=��Z;�틽������P�=SC̽dn�<�G}�����V�=(�/=@㟼&����՛����$��<�o��R7������=�b�(�a��Q��M$��X��<���0򼏞���޽��4�lf޼U�ý(�=�� <h��� Aw�V3=�¥�Ω����;��;9¨=��a�T+�<`�N�� �<��J�d/��	�<��½@/};�����jE< ��;�$�����<����p� �P�U= ����3Խ��0��'&���<JT8��ǒ��D��`׽���;K?ངo2�dR�=��Լ��5�8X< ݼ<6����_�ZSr���< �`�DO��@�z�@�!;j�� �8�@��;�<����O�<WH�ψ���H=@�����<�<P7^<��;��`<������
"=�X�<(f[�Db�<�q� �(�hJ<D�����|�@%�;�]�<�qԼ+��0
<�&<lU�@�<\��< -�`I��nڼ ��>��PJ+�0��޷9��/̼�=�r6=�����a�0]�<��E;p������; �}<0��<h�<a-���}�@����=���;���;�����;D�˼O؄��.L��JC<�.�J������d9= �߻P�M�$�=8v?����;'Y=f J��9T��i�2��o<\ʼ ��;(C���P�8�<��=(ھ<��;pK�;���<�৺�� �G����<$ �� �:�$�� %g����<� �=T�=�W�<Hc���4#=����}n=��<&�0˻���; -h;>{�= h���[��vI<p'd��B�<�k<�C�a�E=hԢ���=X.��~˻����>��T9��
�<jq��i��,��<��1<8�0��U� $.�����,c����ɛ<��7:��<�.$��˽ 9�ɍν�|��A=R0��@��;0��;8���,$�<��=�l~�<`N�=���� !y�Tt���+�:�`�R޽�fz� |i;Y��=�i�έ=���W��lf��f(����<��ӽ0k �`h�h�<�ը<���<ܞ�f8ԽH��<@[o�0��<�E'���)�ֽ�W�=��}���u������hi<�F\� .92�[����H�)�.�:�y�(-<�u8;zm��Aw��k�%N���UF��9J<�=6G;=�����#<���p�<�y1<�+]��w=S��8"-<�ʲ��� �������=�<#�ɽ�Vj�h��<���<$V�X�x<,fi���9�����٣���J[�u���ɻl���9g�n��=�}����ü@��;�Bܼ��B� �¼Ȕ����r����;�j"���� �x��C,<�X<��B�� ��; ͙��+��=��'�g���-<P�'��ȟ;H�M<Tɡ< �; �?��0���6�N0=�<�PǼ���<Pjռ��s�T~�<$��(n�� 9 ҹD���$���8����;2D?�P]<���<`�ջ0��;V��(�O��y� <Z�4o�>b>�,�����=��=Zj����7�=��<��@�; K���.+<��=�)/ϽL��`?����*=_���;䍌� xt��$��OW���H��8�X��*"�0X� ��K=H1��޹����<P�
� �9H��<>c����\�м��y�P�;�y���,<<0�2�����Hj�K?�=�7�< #���H	<h��<�s׼\������+�<,��8=�>��W�:��ܺ�h�=��+=P��<��Ǽ�8ռ@P=�ж�|nt==�<<ܦ�h�Ӽ�<r��P�<2�=����g��@����J����* �`�,��o�<F&�=� ����=59�=����IX<��H= � X�:��C��J�<���=R�=�:�<H7g�Px5��;L'�����<�J|;pӁ;�;��D\=�*���td=��=^���`5�= >z� ������=��=Tm���O��ս̞�P��<�帻��<@C�:v�y���=���<rҜ����=�b0<$�	�X���<�8�=ĭ���
�:h�����>����f�;���<�±�PH >�!ɽ�m��hu(��/��ֲ=f�P� �9:2=�v=���<ɇ�=$#���L�<�����4�dz>��O�y0�= a�;�ѐ=x�'=p	�<}y� C-�Ү< z<p[���|�𽈼����ߑ�ͽ���{�� fJ; �|�@�׻�ή����<f+c�ԡ���A= �@�(�<��>�=⮄=����c߼ē�w�)=��<®����=8�`�� �= ����p=hQ< ��;��n=H��J1=��o< �"��l���E3<�<�<�
/�E����]=p�󻀟�: w��1J�#y��@���Ђ���F���6E� �к��t�>*g�m�w=�hh��CK=p����y=�pټ��<@���+�<���<����Tb��v�=�v����=$�<$M�< <�6<�#=�/m;r�>�j?=����`���Q=��=(��<����ʬE� �a��8�P����b�������6����=$��=@F˼��D� ԅ<���ʴ!� ��<��s<<�$=D��<^�R��&�����=�4*=t�����<�=0��<���.6���<�ݣ<�9#= �;���=&]� �����t=T�6=�
�;���=>"�/刽�wA;
��t,= h޹R׼`�s<0���P=��O=���`c�;@��:d$�<�F�<����ɼ*s=��; �� ȧ��� �=��6=�7==���t�hB2����=��Pj=�u+=Wu<` =�>=���;:��=`&Z< bz�t�� %d� �:�?	�У�;�$2����=p��� \�8	��=p/��P�=�-�<`��;`p��X�M�\��<�Μ=�=�	.����X�;��<x��ҙG=@�.�Τ0����;�n=<���h*>0�<^ٍ�;�X=��U��y�s��=`�;�D3/�w�\�Խ\Ӕ����;���@���h��j�X�^ү=��<x�&�tC�=�9��
B���ӓ=�(�= ?x�pN�; ���*�=|]z�����a����[�>Ͻ�;!�2�Ih����u=b�2� {�;��%=�/i=z=��<=4#ڼ(`���K��|�;N����{��F�2=@A�6��=�=쌠=����<�>�,���ƥ�=V2$��^Ǽ���PTV���T=:mO�X����<�L�<�߼P{� s��J�?�֙M�X7@<��e=��`=��=�.=+In=ZJ5���<����/=�2���2�E�=�Z�<�F{=�)A�`=�F�;�<�=�a�H�=P���)��|μ�H<��)=č��}��p�C=�; ����/��R����ʼHz<�����E<�'0� �:�#�<V�H�}ք=N�3���.=@�B�|��<��s��@J<X�r�즜<�)�< r��I4:�v�<��8\�<��<D�<(��<�ݶ<�BV=@�㺤���W.V=(�i����;	�v=X�<�Vk<�-���k�E <Z�3��W� 1����e!�Յ�=���=�������;����d�߼��:=�3�<�%=��<}6��G#�X�V�-�=ϵ<�w�;�[�<�lU=,�<ׇ������!=���< �<�};<f�t=��L� �R�R�=)vk=�<�i�=�:�^B#� �z�j���9=亜� ��䩃<�_����=���=�:-<x�=�X-<`��<��<8���Ǽ&a=P�;`a��0o�������d=I�m=�7=`��:�E���Hl�= ,:�ph=\��<П<�e<�u=�A�<GV�= J�pM�<�Sb�X�ȍ <���xHi=�:���&�=<)\������W<�(b��(=�3��`�<��-��Ψ<���<$:�< FS�:���½�,=���<`��9&=r�����P�3= p�8����m&>�?4�<lL�,��<@��:���̋�=$|1�Ϲ;)Ҍ��4ʽ0���8@t��T�:dDȼ"�*�݇�=�J�<��M<���=�瞽���;�餼���=��<�΀��a4�@WڻRB�8�&<�ݎ�����=#��=�½��z��.F�%ٽ�a<����a&�4 �<�Y=���<P�{<`E��X̥��o��G`=dD����ѽ@]q;|��^+=n໼)>��ۼtz��|�Z�i�>�4g�Z��F��¼���=��)��ڼ�y�<���<��7�T��$�2�zI���G��&���[�=v �=X���+3=!"=!쐽Cir=()1�0֏;>Խ�����=�J=.�=qh�� ��:�.$���;�h< 4Z�9��=�ZT��դ;�և� =�<��B=��+�����er=��D�`�ǻ8����Ƽ��
���<�f��x,�<��9��@�< *�&vw��t�=&�4�=�w�;p�~< 㥻P� ��d:�<_�<��h<  L�n�= xM:9*=��J=O==BQ=�p����=�� �^�	���h=�7����<�g�=���<�,�;L�� n�H�<�b�(�b��z���:�vJM�rZ�=C�=�[Y�^����,)S� ��:jxD=b�?=��j=4:�<
x�L]����;�C�= 5=�ø<0#=�Sz=�ϱ<����h-���C=�X(;(U<D�=�l�= �P;L���� =̱�=�a<�:�=����V���t��ݼ�6C= uͼ��U���?=�x`��e=O��=8F�<�_D=@ע�l[�<��=��� 4Լo4T=(�]<�懼�����O�Ȩ~=$��=VG!=��ѻb�#�d)I�@h�=@̉<X�=�A=`|�;�M�;|Ȝ=m�=�9�= ˓�H��<r(��/�T�<�9.��־=���<X�\=`м<��N/(�d���pG�<Mt��  +���o��E�<���<����䚼�?P�P6Ža�$=@�<܇$��Հ<|dt�F?��	^=�0� ��r;>�ֽ4M)����'�;)%���I�<`ֈ��� =�I���\��@ 4<�.c�.I���� <jV�|z�d��<�. ��/����<򨚽 g�;(A�<��=�[���Z��,����b����`ym=Hyۼ�t��r�>ks=�_{�����6���K���ݼx�ؼ�38� �r:�|]=@�5<����+��S��9��;�=�� �Pѽr%�����y:��w�I�>H�<�q��Dw�;%>e�~
P��V�ʼ1f�=w�`<��;��;<�G� ���틽n�8�u �l$	��be=���= K޽*-=(Z�<sp��aW�=�1l���7��w�丹=(��<���;�s潸I8�辏�������;��<�6�=psp�\��<��'����<�B=�ڷ�����n�<|�ļP����1��4��o����<xln��5�<�Ur����<\T�������>,��L�=�:�<h�"<@�?;�m���:/;X�K<��B<��=<ټ�1n=����hj=b��=�=���<��*�`��=p&�X��MC=TD��Φ=���=�<�<�0Ի U� `<���<�����ۼ m⼆E{�����
��=���=mץ�.?o���ͼĒ��huf<�~
=߲A=[�=@=�n���Ə��#k<���=X�8=�E�<O�e=�Y=���;�N꽨)l<�=�nȼ�� �&s;=	�=��<(�����U=�՝=`؀<�գ=���W�������μ��1=X���T&�)�=��{2�=�<�=%=�4B=��ּ�r<�=�����g�JZW=8�U<�M��N'������AQ=k�=p =����ji-�4㕽*Z�=Ѐ�<���=�-i=@D=�`�"�h��=�t =�#�=�ˤ;P��;����=��/I=0��;>ͭ=��m=��;D<+�?ր��@�� H��Sbڽ B��n�s����$�<fh�h���lf�6v��Pe�<�֫:ЗF��������b=�t��<�D��:�<�X�=3����ݼп$������R��$��*c�N�	=a祽�v��p��<d^��PA��`,/;�L�dX̼ ����
E� (ؼ�N��'���ؒ;~�b=��=�:t�̒��lh��Vf��<�rZ�=(|6����/�>\F�<�e�6��L�������lN޼`�����}�&��Sy�=l��B��p�ϼ��ͽ &d:�f=����E����½l9Y�n3�_ǽg�=ܾ=����l�,^�=��L�`���;�8!�K{�=����L)�<�,����
<p�%��� ��pL�C{���ڼ���0[<�yh=h��F=����yY���1�=�Jy<A�ٽ\
�@��7�g=��Z�K �S,޽�۱�U��mך������=��=P�E����<䣎��aM;h��<PͽM���й�;d���D�'���*��T���u��~;@��:��;`�j����<�ּC�ڽ6��=��';�=�l%= 0P8�sr<L4��`��@�)�H�<2�8=�U��N�@=�\�d�<SEo=���;x�}<�B�ص=Y��Z�1�h��<P��;�A<�h]=~�<�W|�@������<��������Ǽq���Sʈ�k9�=�Y=5J����Y� ����Tz��X�< �<�Q=-w=�z�<C���>o���<�rj==��?=Xq�<@j�<�ku�ս����ʒ0=�q�T6����<U��=��P<Z7����(=���<8Y<x��=J"�6�V��R�����<>�Р.<��T=�Ỽ�b-=-.�=F�<Է�<��꼀��<< �<N������lX5=@DJ;Ц��p�V��q�e=���=��=(O��0������4�Z= H�<v �=)�=k��P��1:~=��=3�= ��: _~�ص��&e��v=�h�<C0=2��=�6��`D=�,ֽ�3(�~g��V�r���� <hH��3F����< 嶻����t��[H��H<�8��TzE���L�4�<4��� ��SGƽ`9�<�n���ٽ@d���~��d���6��eڼ����<����м7>=n����̽����F���������[����!��0�����<Й�;�_=@}<R��`U
��(�� ���,� �&=�]���<;jS�=��<@�¼�Sѽ��$;[ý��g��k���[N������=L���pV�p��������;�9�;0D�te��>���"s���]��mƽ�Ղ=��=B���,�;�J�=��A�Zi�P�;�Ӄ<.��=`L����<�鍼쫐<�`y��O�� }�:w�˽�ꃼа������`�<��Ľ4�<�V��>>���a=\�.=?"� ��01C��<�Y�
�f�ӛ�� ��,&���A����ݼ�A�=$t���¼x��<`��̨��h'u��$��I��� ��9*m�p8p�$6x�XE(��ֶ�`p����A<��� ؔ�[z8=ԡݼ�
Ϧ= �d�'hu=��X=��?<���Nb�.D��S'���<��[=ZGI��[=�j�П�<�nn=<���P���`>)�$��<N*��KU��< ,���]��sE =�\!=��R� H��@��Ȭ<{�dD�t�*�|���^�����=\�?=[Cǽ������<D�A�xP<����
m<8\=PT���#�f�\���M;���=�[=l!$=���<�A�P�>�䙼`�H<P�1��C��@`:�~�=��*��ė���%=�}�<�X<:L=n�%��췽8S_�F	x�d��<ޤ��iI<<
�<�Y%��<��=3�<�{�;X��R�=@��ǽ����cR=�F�LB��h�Q���e��H����=�G= �O��
%�0ˀ�+rA=@�n;���=s!o=>����-d&=$5�<@��= �; ��.@��	j��4d<<YJ�H�J�pgW��M>�ص���q=�=�=<ݼ�S=#P�=hwS���$� ��ڡ�?y=�7=�,=�-����V����<p������C���T�в�;��l=�A��,��=]98=�ս�Aw=2��Cq����=��<�=���Qٽ��`4v<� ;Ȑ<�Qu<*Ml��8�=�i�=��꼰b�<�b� ���"p ���=t��=��ܼP�Z<�nN��=�=������];PC�<n���o�=S�?��R/��T��G�=�qj�he=�\�=��;�K=LU]=P<����im�(o��k���'-���=��`m=V=�I�=���Ą)�\�<���=D<W���Z��S˼ KR�ƫ}�5ó������/�<P�Z���0��+�;�3�#�˽x[<t=9�0=9P�=�n�<)��=6Yd�8�y�:���Dr�<駼(�C�(�<��<���=r3��u�=�4�(Ol=�w=(>>�؂�= �:�V�����:� )<P� =�(�����R�1=pތ�@`<@�: �A9-4�`:5���� <������O��;\����r=�:��|��<�"���<0�ؼ�廈�s� ��9�_�;����E��`U�<@9-�P�< ��<4C�<P<`8V;]�&=�Ç<|�ż��#=��`|;
Y)= *�;��<�� :�t�@'o; 4�������R�|�������֯=]�d=^�B�6�/���; څ�N����< t��Զ�<h�#<����\���c�Yd�=,�<p�W���=��=pd�<�G��أټ��z;@�R<@E7=p�b<��S=xP��@>^����<�l�=��Q�d��=J���8����<:���|=�;��p�i<�P1�`g�<�O=�%3�\��<��<@%E<�k�;@/
���W�ȿ}=d��< �ٺ@�f;p�ܼ ��; M<�;V=葉���N��1����d=��	�})=9�=��<Pԇ<��=�N�<d1�=��I<\H�<��ѽ��[����< �B�]�=y�4� >v���������F=̹̼��^=��b=0�b�*�?��zL=TM���<0R<�b+<|,ʽH�|<�5#=X�3�8S���~�u��ɪT=+'=�����E/>hvW��^۽�\=�]���#��=�������c��.��B�p� :;|��� z�;(�f<��"�5�=���=��<�G =����l<�=n>�=8�a����<d�r�RK
= ��º��<��,; �=)�8R[�ǈ��[��d�<�r����<�=tԼ��o=D�<�r�\��z�b�H��<�9�Ew���J�=@Aм���=Շ;=�">i���p�@�P�ݼ��>�Ʊ���,�r�&�ήI�u�'=2X����P�Ż0��<��Ƽܷ��tBQ�l��)����N��m�=��=Sj=��=��=98��$��<�}�`)=(����
�~\�=t�= d�=�*���ˋ=Zch��ʒ=�20=��:���>|���M������V<�X$=�8�<H���=4ؗ<��$<��p<�I���� <��<ގ�pW�<��9����Dp=�"���=V�\� �7<XG[��z5<ܹ�� v:�뀼`�n���s;p����	<���;@�8<p/<80�<�<��=���<��W=p�l< ��c3=�w��(�8<��7=��O� 
�< Qp:�a��0)�<�l�;���<�Ϡ<����<��b�d=�o=�x]��OD�P����έ����&2/=HIq<@��:��R������Ҽ &��yԊ=�
�:@�o<�U�<��F=���<�������F=�<�=��<ȧ�<`�� ����(w:x��= �W9��=�g��n�T��<�a޼��;=�27��Տ����<PE�����<��\=��j<?A!=L��<��C< �;�ݕ<@N�o�d=���<���<@t�0�h�8�<��<T�;=@$��D ̼`)��lQ�=�Au<8v�<@��;��'<P�λa�M=~eP=�f0= ���WE7=�a��\"C�X��<��F����=,-%�uz>�����!�81p�
-�?��=�<�?i�[T��N�=���p����ּ.6���2�y=|=D���TT�j%ڽ�	=�E��=0��;X�(�yz>�����ȽJu=1�����)�=�1����A=��ٽ)�	���\��������;P��;����|�=.��=�"�<���<�<8��<�q%���/>�\=�3b�x<�`���Ƚ��L=h<�@-�P�=^�=j�:Y��2Ͻ��5�����G� ����J�=01�7�=p=��\M��+ռ��0�;��=04�Pν(!�<,�
�Ģ8=@3�:��U>z��Xm]�f���B�9>��\�0�����Q͋��=. �.�1�`�;L�<��1� �
:<_�Z��ܽ����=���=�A��X�O=/S�=$���ɢ= _��P=��@�X��<䩿=��=��=���D�<���� I= �<@���J>L�Z����X»8��<kl_=`�������!�< .�;��� ��8PWU� ������<,���Dm�<�wm�r�<hq<<-˼���=�V�c&=�-<��<�0��V �p,� ����2;\4�<PK�;��<�K�<�!�<��K=i�=)=�I:>u�=(yd<�7��a= ���~�= �y=�<8a<�B����<,�!=x\�H]�<Dr�<,(����=Z�==
���4��4�����B��x�@,E=W=���<@?";�ˇ�Z'0��x�<�h�=�<�4=�P=l�r=H��<J�E���λG�%=���; ��<�X=��= =���f'��[<���=��<���=� ݺ�@�:<�~��:.d=t@���<�M=0���>=@ې=ؔ�<:c=��J<���<l޾<8���Z��|m=6@=X�<h���4�$�@u�<{�l=Q�,=�lH�Tc��􂄼U*�=D��<��u=|<�<ظ!<x��E�=�=[�J= k;��;=<Rf����O	=T�"��>�7���F�=�#��b�e���{���)��l~=������K����~>�.��w��81�Up��?�$�=�O�<�R��5���ǽ>Eg��Q�=lQȼ�J.< 2v>r'�����ṕ�L༴�Kf=~桽.�r=_G߽����e$�Ԍμ"*�����;8z ������e=�y=Ps�;�����*�@�K;z�;0�> ����������n���E�μ�=@s�����@M<>dK�</-��|�p�/F����H�I������d�Q��79=�@��x�/�.�X�)�~
:���鼓�>�^3�C!Ƚ"��h�)�`9�;PD���=>��;��J�ֶ��CQ/>�8a�Ł��H�A�G��r&>|��0活 �;�)˺��l�@�G<�C��v�3��'����G�b4(=�)>8h/�-�P=��=��%�;P�= ����*�;b�y��>D<k�=&��=�h=
!�8b���bd�p��;��<���<3<>�_�+�< 7ջ]�!=MZ=�t�k��t�<@�������|���߼�<���<0V��pE�;І���=��g�S�����=��T��c�=Ϟ=HZm<�FԻ�$� w��mŻ`p�L�=��֩F= �+<�S=�5�=�%+=k9=�9�rڍ=  �n��pR=<ڣ�h�6=��=�ߝ<`Nr�t'��y<<V"=V�+�@��;pG����!�U䆽t��=i��=cٻ�R0��Y�#����� =݁&=�Mx=�c�; �?5���
�<cȯ=��3=^�=Wq=�U=P{�<������<�J=�T��`��;V^4=�t�=x�,<Z`���g�<D��=`2F<[�=(W���ἰ��@s����`=�ۍ����;bҘ=P/��	��=d�=��
=)U=��(��R�<@.�< b��n����=�^�<�Ӽ����t���&�<x��=��2=<� �V��7>��q=x�
=/"�=�*]=��;��&�n�=h�p=��=p��;��=��e�\1��::=8�b���>�9=0�U=��?��oX�����Nu+��=�g}�(�k��+�����=p����Rf����#���������=��<�lFw�����X�C�,�M���=&�e�|�<�*)>:B��ߌ���5�D����v���D�={.=�!���rݽ��:<�|&��ɽ��������.����<P��;��%��5��̱Ͻ$�����<�޽=�E?����,͓�(O<��}W����=�5>��)����=>�W����C�,�{����tE1�/	���]q������ef�=Tr���@� �޼퇧������=4��%�������tbG��j��'G���7�=�x@=�Y:�V#��zQ�=F�:�|�����8]���> <���<��;�����Pm� t�;tR���cY�6�!��5A�0А����=��2�gvG=-
��n�����=�l<���^��Ц����==�D�f���FR�v
�����sI<�'�=��=8!�|��<8�m�ؔ�<���<X��z�z��w�;0S��*����.�@o�� zX< ��; �ٷ���=�3��m���c�=����,'z=��U=P��;�_Z; (]� �	����0�s<�d!=!���A=��*<���<Ö=��T<<ؒ<h����r=�sf����li�<��;t&�<�k=���<]� ���4:<��<&yL� �ܺh3*�>�n���]��$|=r=	�ƽ����ͻ��e���'<��u<�=46= ~ѻBqj�z�;�4�<W|=9�7=��v=�~�<��<�Ռ;u���@R�:(�C=hw���Қ�䙗<��=�c�;X�x��*�<r2=���<l�=���C�`�J�b�"���:=@�f�Qo=%.q=(�~�ę=�@�=���<�y�< ��� �<���<&g�����~�z=�A�<��R��xO��:g��ё<4��=O,.=�7ʼd�ټ pG�N�+=H��<�ÿ=��=����@U^�=�= �5=�ԡ=�@�;l��<�i=���L�H�Z=P1<��=DÆ=���;����{(���l�h>�p�\<]�ʽ 1�:pN�J=8�:<d�Ƽԟ7�qҀ���N��y=�����`�������;����R�I>���v7=S�=7�
����d戽�P���������(�ۼt� =����6��� .�<r����1Žp���V�8��Лa�8S	���伿aŽ��+�� *�c�<��5=T�~�(z-��ؽ�8	�fA?��Q�=4�e�\O/���>�x%��B̼`����p�� x��>����et�:���2�=nq�$�B� _���@�;�r=�^��:4������B���P�����V�=y�a=h�@���o�'�k=�V��]���)�L����=�P�<(�< vF�@�(<����YE�Ģ��,�������� �Hc+�|��=���>F!=Fo>��2����=L��<�v���������*=(1�����uƽ�m��|����� ��`γ=dy�<��T�'�<|�`��; g��#Ž�В��ݻ������&j��Љݻ�������X}�<�5��m]�Ad= ����G��J�= 5��7o=b�=�:@<��6�Pym���8��c�H�<M==��+�WZk=�����'�<�m�=�g�@5���ټ���<HKڼ"��84\< �)�h�$���5=H�;=�3���햼�PI;��<.2`�lEļ4�7j��"Vy���=�h=��vm���<��6��D<PE��୭<�g%=HJǼAE����0��?
<Tm�=��U=na=p�.<��컰� �=��0p&���<��-���� 0���ķ=�#�(���= ��<��<��U=�XּI9��@Ո������=@-����*=��=����i�<�P
>�/�<�t�;��ܼ��=�l&<L�ν���rs�= έ:�߳��+e�Hsl�h�d�L�>� O=�㼒m+�(�Y��=��A<�@�=�j=���� d���=x�<To�=P��;�?D�:���8w��p��<X�����o<|���>����=b�F=�
� [�<B��=���#���=Jn� ��9`��;2`J=�3���r���#=��/;��L����
^N�^�A=h!<=��8��=�,�<�H����=^*�;Y���J=p=�<����OO�,x���������*�;�\�<L��<6<����=���= <��Ǽ�$����<�l���=���=�������:��^�1O�=���K����(=r���\S=��t������۽�H7=�Ǝ��n=B�=�j��Af=�ٵ<J=$=���������#�X���f��X��=<��D��<���<�l�=����:�p>�;{�g=��;x㓼�,��Mм�#�5|�������S���=ܚ��`;�8Es��b��8 �� ����=R�=⩁=�~�D�h=�~��`TL;��L�,�=T�ɼ�i� (h��h�=�j�=*IK����=�ס��|�=K�c=2ҋ���=X�޼`b�O=�x<���<�7��X��tb=`_�5�<�@<\S�<j��𻎻p�Ҽ0x���2R<�j�<(�q<�t�;K���}�;�P<<�=�ռ:�@�޻@�o� #`�@��:�f��X��<�傻��<`	�< �l<�Z�;)C�0��<��<@Sٺ�w�<���8<�8�<p/һ.A= AW<��9�H|b<\*=��; _ܻK<�F����=��=.8I�FY�`�;@T%������<����Tt�<@Tͺ;�����S���f�=|��<H�T�p=X=g<�N�<Ri2�����������$W=D�<J�=H�J���ļ@6d;iE�=X�c���=h[Q��Ѽ
M�=�����<�Ϡ<��ϼd:�<�c���=�B�<@��:�ׁ<`��; /<�e���o;�-�;�hv=�D,=�<�O�<(m�HKݼ��ۼ��{=����Jż�5<��=�Ä��q=	5=��< k����<���<�= ��<	!=,���вy�D��<�����=�B��c02>td=�l�E�XlD<��*���7=<x�=��/�8���^�=WӍ�fA�����|��<��н�\��G=``=;^������ܽR��=�`;=�*��V$> �׽�;=J�E�#��0σ=$ ��@.�<�����b���͠�X�W�(�p<�=�sE��/�=��>��=�׏��� ��E�<�=�*�/>fN�=��k�t#�<���l<�!1��*� �<ؤz<�@=m��T�$��d���0#����6"=�+>Z_Խ]�r=�m�@��<]��������<X���g�0�=����e=�[�<�?9>�@ҽ���-���=������D�����y��i6<B�v�Q1��t���䮼<0���K<�ߤ�Vc��K���ɼ%�=xC�= ��:@�;&~�=����	=�,J�ˊ=�Խp:�<��=n��=x,�=U���8�=s�սL;�=ܦ6=fݖ���0>�;T��hż��<Ж�<���<���< 	H<�b�<���<��z<�P�<Lj�<��< ��<&��0�q<@p<�U[<a==��<�ۿ;
�t������=/<l��< ͩ���@�O�Xs����$��(,<8y<��;ء�<�)<���<�"�<��<���;2L)=��<�+�<(
=��w�0"7<��<�e�<,=�J�<t��<��<�&+=���<�J�<8�< ��:��C=584=.�I�L�����q� x�8��s�= ą��b^��%�xt�<ؠ� iN;��=@�s;�jA<��C=��<D/�<�4�@�G�0��@i<�s+=���<�u<<�䡼X^�������=(�'��%`=8J�`0<l��=�0O�E%6=`�< �;:l��< ��7�
=,p=���<.I=(p�<���<0��0%�<@<v<�g=W5=m�9=P�<`��;h2���C���m= �8;�mT��G�<�FC=���<$��<�7�<`9�<Pq|����<gzS=�`�<|`�<�ER=��x�ȨE��c�<����L�>(̽��3>p�l���O�Z
B�^�6A�=��<=N��q\���,,>�'j��^ͽ@�,�aX��齁�Q=(=缨����A��M��		>�`�<��O�]��>`��o�� ��<�XY�t�.�dr=ܴ����=������~�=�4Kļ�[T�0٣��9=��<_&�=�>��5=تq��g��ex<�'�$�C>���<���P�W<�Ͷ�y����=�x���Sc���>�N<������5��\&�x\�_䏽z�n� ��� f�=�\��3�<`�f�t�������b~�$�
>5\R�QϜ���<��(O=��@�2�h>bZN����㽦6>�y������>��AϽ8��=�����Z<x5<$?Y���<��֨R�����lb�	�=9.>����~�<�W�=V�@���=I�����=F�Z�t�=�Y^=#�'>�f�=
e�`��<�7��7�=0r�<<�S�D8�>���P|	�� <�=>}%=E�;����T)�<0��;P�*���<,��<�i1����<���Ȇ�<O�; ��<�=@V#��N@=Z���<���<(��<lX��@�ûA#��&���,�$��<��;L��<h��<��<},=Z=$�=�����=�ȶ<@^;T�K=,���h��<;�g=��<���<P٤��U�<0}.=L��<�#=�v�< u�:X�n��h=$��=�z�J�\Hͼ$��@x=�65B=@Q<��<`� @;��ۼ���<�"�=b�<0��</�p=�v+=�=�<j(�@�:��;@�];��<��=s�=�����`r�F.�= T��=`�����<�M=0�R��z=`ge;8X<tMS=�ש��|T=ډ=��%=��;=X'�<�*�<@�R<��u��ʦ;Y�=�5=4k�<��C<��xNI���=_5|= ���5?�@Q<�~l=/�<	rh=^C3=���<�����<g=�m=�)/=�<�fK=PI#�|M�hg�<t�f���*>�����p>�꒽����휽c�O�k=�s�<�:��������@>Q��BܽL)��Ƚ�ӽ3W�= TS<�_[����k���q�%Q�=�3&<�o<���>����蜽hx����/�` 6�Tl�<(���@�=�A���h��\���g�������j�<$�=@�=���=��<�y��T��Ɍ< ks;�Q>�������E�����uB���>|8�L ڽ!G>$ｼ#7��0༴G�X\�'Ž�5�L�T�B��=_e��t,��]��� ��/�����5>k�A�g\������Ȕ�8��<JGE�7>��*<��T�����">卆�я��pǸ�ycؽ� > �; |n;��=(�%�02��Ȭ�<�#��`m��$��B�����}=D<>��_��]=<��<�AD�Z�>H����]=�����9=�u=]f>�=:� �@W�E���==�e�< �P��Pt>����e8�,ρ�G�&=s==<<��l����s<��������¼���;T���h��<x�ټh�<<`f��]�= �b<�[-�\y�=��{�٫7=�h=Xĺ<`�� ���k!� � �߹8��<@�8���==�u�<�O�< �_=f�=̰�<��3�:��=�T<����&pE=�v��u�<��=�,m<@��<��żP��<#�"=��m�h~�<P��;����|Aܼ�=f�=�=����x��|<s�`�$���E=$��<8�=�[[�g��Ξ?�А�<��=�p�<�W=�G=�36=���<6xs���>;`ʛ<����`�<�)�<�R= 4>�d�V���<<���= ���Ϟ=(W_� v#��|<TF��=zl=��T�(��<��l=�$x��]=�w�=˚'=� =`V�;�^�<��<�o��"�=\s=�x�:�3�X�� {;V��=�Mt=L���l���P�V�Wq=d��<v�=0�O=�+<�H�����=��G=6�=P<<�=�3�@l0�T4�<�>���W>���ڶ�=�7��lmx�������=���= �;��"�*1V�Ј>�Զ�����\B�/-�ރ���ż=������m���(� jĽ<{X��%N=��=�l{�<�{D>��t�e��|s����@��������ς=�᜽`���?�;�R�:P����%� ��pm�<�u�<�k=��D;�׀�ս���$<� A<��=|I�4���	��P�W���?����=��x�*����=>.�F�]��W�ߗ��|0��ٖ��y�@^���f<�=Wӽz����;��ف�0�»��>
���f:]�i���,s���+�����Զ=�}l=pf�;�
ؽ�2�=2.�N!1��+��Y���V�>��<l7�<�=@���6ꅽ<a�<
��~iX�D�`qR� ��c�!>f$a��HB=�9�����=$�������Ud����O�R=���=�&�����@ؼ�f��9���ݝ<s$C=�>p�/�P���ȅ�F�<P��<ଲ��߼�; ��:�N��@Q�@Y�<�!k<`g�<��%<@�Z<�"�;Gr#=`��<���V�9=x���<�.+=x�<�\X�F)=8�?���>����<�< :\<lo/=�(�<@_�̅l=��<xDh<�J�<~)=P	<�P�;���<ܻ�<p�;kz=���<��; ���D�<���<�����<䫻<�开_N�L��<l'�=����Hi��=,��%μ ����=RP= �������׼�ռ�9�<�ex=���<B�f=Z�;�e�<�H�<�V!���Ҽ�u<=pM�; ��;@(R�q-G= T>��Jʻ�^<"�&=�/<�s�=����/ּL9�6�h�a�7= _k:}~�=|��<@߹�XAV<�L�=��=���<��A<ha�<��
=`�)�,|���6�=�OJ<�XB������xV�8k
<�H�=vQ]=�dֻ茵��dz��>=��<0V=��<H4�H`I�n M= �
=��=��ѻ%=y%�@�U����<-�;��=��c<���<xz*���M�ov��9L���<�>U���c:Q�t��=@��V�l��CI�3Q۽��N�pH�=X<-�X�H��d��|6� i%�8�J<���	)=`s�=D���8��j蝽�����Ǻ� �G�yO5=z{�D����J�;L-��`���~�S�$]ڼ@��:�2[;�\g�P켼)S�� �����@;�K<�~@=Bm���e?����� _7�b%�3��=���eǽ�z>Y�Τ�� %�C7���B�H�����7��{e�L��={ӽdᒽШ
��멽��6<H��=����4��-�ѽ(�b��K�S㺽�=G��=  8fʢ��)\=� ��f���%ͼ�̔��� > �	=���<d��<�G ��,���;�����L���H�'�L���(�=6/��� =�x�S*�����=��������L �ֱ�Z	= 7;�U��K���>P�K���"���<H�=�X=D���0�ü���� �;@�ɻ���Z���q�p���h��8�g�\3�< ��9Xz<FJ=x
� ]��fK=���;-�� V�<'4<$Z�<��R= �/;��i��a-=�욼䓽��=��R<@���%+=8<x)g�EPP=�.I� r`���<`R< :o:`�;`��;T��<x���D1=�&= �����;�s<�Jv<<ԼPz�;ܼ����>���K�<��z=��|���%e<		� >�9dҳ<�<0����`���Q��[���Qi;�Z=T�<��\=�����
����;:i�ں���'= ��H�� ���,v=H�c���
���<HsP<P�<�
9=:׼^T��{��C�����<���;�|�=pP�; su�h�
��q�=P`	=��û�8�:�G=,v�<��z�D�׼C%�=�0���J"��:��.� P9��>'*W=��<�t�꼔䨼,Z=ؾ/<�Q=̚�<\ʼ`W�"�=ȿ#<C�=�����(�<*
dtype0
�
siamese_3/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`*
	dilations
*
T0
�
siamese_3/scala1/AddAddsiamese_3/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_3/scala1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_3/scala1/moments/meanMeansiamese_3/scala1/Add/siamese_3/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese_3/scala1/moments/StopGradientStopGradientsiamese_3/scala1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese_3/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala1/Add%siamese_3/scala1/moments/StopGradient*
T0*&
_output_shapes
:{{`
�
3siamese_3/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala1/moments/varianceMean*siamese_3/scala1/moments/SquaredDifference3siamese_3/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_3/scala1/moments/SqueezeSqueezesiamese_3/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
�
"siamese_3/scala1/moments/Squeeze_1Squeeze!siamese_3/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
�
&siamese_3/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_3/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_3/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
ksiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_3/scala1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_3/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_3/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_3/scala1/moments/Squeeze_1*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_3/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_3/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_3/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
c
siamese_3/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_3/scala1/cond/switch_tIdentitysiamese_3/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala1/cond/switch_fIdentitysiamese_3/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_3/scala1/cond/Switch_1Switch siamese_3/scala1/moments/Squeezesiamese_3/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_2Switch"siamese_3/scala1/moments/Squeeze_1siamese_3/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese_3/scala1/moments/Squeeze_1
�
siamese_3/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_3/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_3/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/MergeMergesiamese_3/scala1/cond/Switch_3 siamese_3/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_3/scala1/cond/Merge_1Mergesiamese_3/scala1/cond/Switch_4 siamese_3/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_3/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala1/batchnorm/addAddsiamese_3/scala1/cond/Merge_1 siamese_3/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_3/scala1/batchnorm/RsqrtRsqrtsiamese_3/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_3/scala1/batchnorm/mulMul siamese_3/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese_3/scala1/batchnorm/mul_1Mulsiamese_3/scala1/Addsiamese_3/scala1/batchnorm/mul*
T0*&
_output_shapes
:{{`
�
 siamese_3/scala1/batchnorm/mul_2Mulsiamese_3/scala1/cond/Mergesiamese_3/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_3/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_3/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_3/scala1/batchnorm/add_1Add siamese_3/scala1/batchnorm/mul_1siamese_3/scala1/batchnorm/sub*&
_output_shapes
:{{`*
T0
p
siamese_3/scala1/ReluRelu siamese_3/scala1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
�
siamese_3/scala1/poll/MaxPoolMaxPoolsiamese_3/scala1/Relu*
ksize
*
paddingVALID*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides

X
siamese_3/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_3/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/splitSplit siamese_3/scala2/split/split_dimsiamese_3/scala1/poll/MaxPool*8
_output_shapes&
$:==0:==0*
	num_split*
T0
Z
siamese_3/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/split_1Split"siamese_3/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_3/scala2/Conv2DConv2Dsiamese_3/scala2/splitsiamese_3/scala2/split_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
�
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
^
siamese_3/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/concatConcatV2siamese_3/scala2/Conv2Dsiamese_3/scala2/Conv2D_1siamese_3/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:99�
�
siamese_3/scala2/AddAddsiamese_3/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:99�*
T0
�
/siamese_3/scala2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_3/scala2/moments/meanMeansiamese_3/scala2/Add/siamese_3/scala2/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala2/moments/StopGradientStopGradientsiamese_3/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_3/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala2/Add%siamese_3/scala2/moments/StopGradient*'
_output_shapes
:99�*
T0
�
3siamese_3/scala2/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_3/scala2/moments/varianceMean*siamese_3/scala2/moments/SquaredDifference3siamese_3/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala2/moments/SqueezeSqueezesiamese_3/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_3/scala2/moments/Squeeze_1Squeeze!siamese_3/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_3/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_3/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_3/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_3/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
(siamese_3/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_3/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_3/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_3/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
c
siamese_3/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala2/cond/switch_tIdentitysiamese_3/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala2/cond/switch_fIdentitysiamese_3/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala2/cond/Switch_1Switch siamese_3/scala2/moments/Squeezesiamese_3/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/Switch_2Switch"siamese_3/scala2/moments/Squeeze_1siamese_3/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_3/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_3/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_3/scala2/cond/MergeMergesiamese_3/scala2/cond/Switch_3 siamese_3/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala2/cond/Merge_1Mergesiamese_3/scala2/cond/Switch_4 siamese_3/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_3/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/batchnorm/addAddsiamese_3/scala2/cond/Merge_1 siamese_3/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_3/scala2/batchnorm/RsqrtRsqrtsiamese_3/scala2/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_3/scala2/batchnorm/mulMul siamese_3/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_3/scala2/batchnorm/mul_1Mulsiamese_3/scala2/Addsiamese_3/scala2/batchnorm/mul*'
_output_shapes
:99�*
T0
�
 siamese_3/scala2/batchnorm/mul_2Mulsiamese_3/scala2/cond/Mergesiamese_3/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_3/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_3/scala2/batchnorm/add_1Add siamese_3/scala2/batchnorm/mul_1siamese_3/scala2/batchnorm/sub*'
_output_shapes
:99�*
T0
q
siamese_3/scala2/ReluRelu siamese_3/scala2/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese_3/scala2/poll/MaxPoolMaxPoolsiamese_3/scala2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
siamese_3/scala3/Conv2DConv2Dsiamese_3/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_3/scala3/AddAddsiamese_3/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_3/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala3/moments/meanMeansiamese_3/scala3/Add/siamese_3/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_3/scala3/moments/StopGradientStopGradientsiamese_3/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_3/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala3/Add%siamese_3/scala3/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_3/scala3/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_3/scala3/moments/varianceMean*siamese_3/scala3/moments/SquaredDifference3siamese_3/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala3/moments/SqueezeSqueezesiamese_3/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala3/moments/Squeeze_1Squeeze!siamese_3/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_3/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_3/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_3/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
ksiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Nsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_3/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese_3/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese_3/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_3/scala3/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_3/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_3/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
c
siamese_3/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala3/cond/switch_tIdentitysiamese_3/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala3/cond/switch_fIdentitysiamese_3/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala3/cond/Switch_1Switch siamese_3/scala3/moments/Squeezesiamese_3/scala3/cond/pred_id*3
_class)
'%loc:@siamese_3/scala3/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_3/scala3/cond/Switch_2Switch"siamese_3/scala3/moments/Squeeze_1siamese_3/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_3/scala3/moments/Squeeze_1
�
siamese_3/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_3/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese_3/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_3/scala3/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_3/scala3/cond/MergeMergesiamese_3/scala3/cond/Switch_3 siamese_3/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala3/cond/Merge_1Mergesiamese_3/scala3/cond/Switch_4 siamese_3/scala3/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
e
 siamese_3/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala3/batchnorm/addAddsiamese_3/scala3/cond/Merge_1 siamese_3/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_3/scala3/batchnorm/RsqrtRsqrtsiamese_3/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala3/batchnorm/mulMul siamese_3/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_3/scala3/batchnorm/mul_1Mulsiamese_3/scala3/Addsiamese_3/scala3/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_3/scala3/batchnorm/mul_2Mulsiamese_3/scala3/cond/Mergesiamese_3/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_3/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_3/scala3/batchnorm/add_1Add siamese_3/scala3/batchnorm/mul_1siamese_3/scala3/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_3/scala3/ReluRelu siamese_3/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_3/scala4/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_3/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/splitSplit siamese_3/scala4/split/split_dimsiamese_3/scala3/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_3/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/split_1Split"siamese_3/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_3/scala4/Conv2DConv2Dsiamese_3/scala4/splitsiamese_3/scala4/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_3/scala4/Conv2D_1Conv2Dsiamese_3/scala4/split:1siamese_3/scala4/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_3/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/concatConcatV2siamese_3/scala4/Conv2Dsiamese_3/scala4/Conv2D_1siamese_3/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_3/scala4/AddAddsiamese_3/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_3/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala4/moments/meanMeansiamese_3/scala4/Add/siamese_3/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala4/moments/StopGradientStopGradientsiamese_3/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_3/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala4/Add%siamese_3/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_3/scala4/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_3/scala4/moments/varianceMean*siamese_3/scala4/moments/SquaredDifference3siamese_3/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala4/moments/SqueezeSqueezesiamese_3/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala4/moments/Squeeze_1Squeeze!siamese_3/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_3/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_3/scala4/moments/Squeeze*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_3/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_3/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese_3/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_3/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_3/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
"siamese_3/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
c
siamese_3/scala4/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_3/scala4/cond/switch_tIdentitysiamese_3/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_3/scala4/cond/switch_fIdentitysiamese_3/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala4/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala4/cond/Switch_1Switch siamese_3/scala4/moments/Squeezesiamese_3/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/Switch_2Switch"siamese_3/scala4/moments/Squeeze_1siamese_3/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_3/scala4/moments/Squeeze_1
�
siamese_3/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_3/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_3/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/MergeMergesiamese_3/scala4/cond/Switch_3 siamese_3/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_3/scala4/cond/Merge_1Mergesiamese_3/scala4/cond/Switch_4 siamese_3/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_3/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/batchnorm/addAddsiamese_3/scala4/cond/Merge_1 siamese_3/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_3/scala4/batchnorm/RsqrtRsqrtsiamese_3/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala4/batchnorm/mulMul siamese_3/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_3/scala4/batchnorm/mul_1Mulsiamese_3/scala4/Addsiamese_3/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_3/scala4/batchnorm/mul_2Mulsiamese_3/scala4/cond/Mergesiamese_3/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_3/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_3/scala4/batchnorm/add_1Add siamese_3/scala4/batchnorm/mul_1siamese_3/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_3/scala4/ReluRelu siamese_3/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_3/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_3/scala5/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_3/scala5/splitSplit siamese_3/scala5/split/split_dimsiamese_3/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_3/scala5/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
d
"siamese_3/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/split_1Split"siamese_3/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_3/scala5/Conv2DConv2Dsiamese_3/scala5/splitsiamese_3/scala5/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
�
siamese_3/scala5/Conv2D_1Conv2Dsiamese_3/scala5/split:1siamese_3/scala5/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese_3/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/concatConcatV2siamese_3/scala5/Conv2Dsiamese_3/scala5/Conv2D_1siamese_3/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_3/scala5/AddAddsiamese_3/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
O
score_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
Y
score_1/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/splitSplitscore_1/split/split_dimsiamese_3/scala5/Add*M
_output_shapes;
9:�:�:�*
	num_split*
T0
�
score_1/Conv2DConv2Dscore_1/splitConst*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
score_1/Conv2D_1Conv2Dscore_1/split:1Const*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
score_1/Conv2D_2Conv2Dscore_1/split:2Const*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0
U
score_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
�
adjust_1/Conv2DConv2Dscore_1/concatadjust/weights/read*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:
�
Const_1Const*��
value��B���"��-�=4��<�k =f��� _�;��U�,_�=��ټܳc=!<��6;��C��(�*�z��%��V���
�X�K=p-<���H�(��=x$�$ڕ<����N��h�= ���V� ��;\d=п���ü���<Rd=��<ܑ�<�S=�*�=�Y���=�d�� y�;ˎ=�M��F�;���<�+<�U[���;���<N�&=4���sv�=��<̛�<�9;=�nM;��=2Ň�%�<=X
e=�S=�b=���H��Ƞ��p��t��]����m�:��;�S��ʉ=�`E=������T�?=`�&;��o�s`���=3= :#< �7:��1��t=�ݶ�t�O��м�N=�h�T��<���=�J�1ߐ��|�$<��`�;�Or��ꕽh1M��һ
�V���t=�J=���:�����T=�'�<w�� �=�9ļK��=���<.s=���϶�=���;�֟<{��=�&�兼��=�Kz��=4m��d5=�=.;��%� 7���m]���R��$c<p��l��<@�"���]�0�3�0���з����X���"�8��< ��9���� 4�� �~;@��: ��� $����߼@�:�=ȓ�����LK���x��d�����<(���w����� �����=�
8� ����V��01<��5�R�N�h*2� �g�h=��w!���q��3��@�;:�<<ߐ<<G��U=Z�w���=�Ug�x@�<�:�:�1Ҽ��������;8I�(��<��&=t�������s���c��TT�`�����<��A�L��< A���r�p,x<��X�Z��`= ��n%��ű�D��d>μ�a�.���9\�����X���@��� ��M<0��;�mn�@���v����)�l���I����;x�<H�z���� �ǻ`�����e= :]���F=(J���ջ@[� �`�Pʄ;�0Q��<tb���@az�@���z�; ���@�>��"=@ڴ�g�n=X��<���<�@����9;�tֽ ��<0�'<� l=8c =�*�����P�K����,��.�,�μD_�<��=����y�@θ�%�=P���c�%=��Q�Pe3=p�A=�<r� �$g�<�Y�<PW�<�d�� �r<�fʺțM<��={�Z=��=$�<�V~=���<���<�E�����<�nJ<�+�=4�P=�˄=�!:=�m�<C4N=�=�Gp<��='A0=4��=�|��5�Ư =���<�֥=����� Bu<<��<�X��9��ʶ<�l���<��M=(�<Xfk<f�x�4"2=h:p< �d���� Ѽ �ڻP�v<x�v<�u�<��u������-�X\l<��ѽ#;�=�]3=L��p�H�r�=�����"R=���<�{�(x����<�]ż�������<�=x���%�<v��=�'{���I=��=���.�E=�E�=L=P�ệ�x=�9����V=���=L�<����(�)<��9���`=0O�of�=K�H=@d׼��B�H�R����? =�PW�n/
=���;:q��;�,%��H�'�~�w���x�u�X�=��<��C�v�/��jɻ��<�q1=ļ��2�� �G��!=<�#� ��;R�7�^L0��u漧2d=|�'���i�ЭR� /��ѼG����q�l˼$�#����u	��p��;��^����r���������)�: y<d�<)��T�=�0���:=n�J�/==���<B�Q��b�`Z����O<�fP<P=l�h=D$����2<hv<T_f��E�;XcF���<����;p9������<��i��k.: 8�<�m�@�<� �y��d����Ҽ@!�;!/�P��<�O<�����H=<� ��K�;p�< -��v�=����`��+<��k<,���I�<V���h��`j�;�j��ܩ=P׵��<���@��<𳋼��?��j=$4Q� 7z���;(+����4n�<�ܥ<�� Y�XRc=��)�͊i=�W8<�*X�xP���j����E�'=�c=D#t= �:r)/����� ��<@q<u)���v���=���<'I���#����S�b= �ݼ䮑=́���=����ȇ<��
�8�L����<ʝ�=��b� =���P����H=�L=���=���="�=h9��Ј<�񭺐Ď�p�̼ i����S
>�)�=eC�=�z2=��=@�:#h)= �:�傻��=�o�=��5�l��� 6B;�UC�r��=>g)��o	��k6=̐=
�<ؼ��:=wؚ���w= )�; ����;=:v0��+\=�h<�'�=_v�_�������7�<�,=,+�(�q�U�<Ċ\�06A�� ���=0�e��f��G;'=Y�=���<ׯ�=��n=��Y�����*�=�J���tպ�n�;�"�<�|C���=fc=��R��+�=82]=L��x�Gh�=��߼��; �B<n�I��S�=)uL=:�V=@�2� �%��i<X�<���t>o5�=dB��.�� 3M��qr��o=lO�R�=�H�<��n����0?׼� 9��%���=����<���=�q"=��"����� _;�6e=�+�=J����̽��h<��U=��/��=o@���
�x�b�=<Y���A�8Zr�@����.�����<@ؼ�������J.�mA�B/=�_#���w�8�G�?�<��X� �y�(I<�U�<��l��>�5��=�K����y=�=�oe�  _�l6Ӽ=�M=7�\=`��<�6i=�k¼�DY=��d=�A����.=��W�h�=Jsc�x 
<�
� 
V�l��<ϣ<;�=P��<�%`�p>�;��7=�Cr;�-���p=��D�G��=�Zh<�Wڻ���<����T`�<�!=Tؕ<X��=��S�"9=��q�6n=f�5� �6��/��� ��<�Ἦ�=���� `F�l����2=��ü`λ�"Z=R ���
�hm�=�o=� =/�1=ҙU=P�`���r<�M�=���� �=���d?������dvK�ơ������=���<{��=����^�~8o�{n&=P�+=��彐u�;$C�<`��C� ��;>!��4<L�2�vE�=�.��ft�=
��� l<��ҼJ`:���g<7��=H�O�n==²1�����J�<�=��=��>i��=�:��8�n<�����	��dⰼ����(@���>�	�=���=��=Q��=Ԕ��U<<ඊ��jQ��p�=�3=�T�;�V��()��D6��[��=fQ[�TxJ��@'=��=�l=�?����(=�ʽF�E=c�\tb�P�=�e��&^=�;ә�=�����|GI��F�<��<~���1W< �J=���L(���Z@� ��=:jM�|�����=*֕=�u%=���=%[k=�M�,�*�S�=��P�0!�<�h��G#<@ST�>뻕�A=Ǹɽ���=G�=��*d�"
>V+��?Ҽռ��Ž�q=H��wFE=l��0Z��>�=Э�������>NX�=X�K��:���]��H��S�h=�$��<l��<�]�7�׽t�����dĽ0�R���=��=�=p����������<�A�=���=t׼�dϽ ��<{�V=�h�i�b=ij��q5���(�X��=���������B���������3K���`1��b0��a۽��;�н�r=h���&� �q;T�<Vs��;���\�;��I<Gӊ�tF">@��2�<k@ս�\=�=��<��[�,"��x�=���=�l<��=0|d��6�=(�=넽�|=O=�?�<�V�����< �	��ns��D�<��=���=���<�i���E�<6'�=���<`�����3=�WR�h/�=���<�$]<���< ���t��<��6=?�=m��=������K=��S��6^=~�1�0X��X0��0K<8 �<�3�糓=�̧�x�r�\����g=�����K9�rs]=�h��f�3� >�Z�=[L=yoD=P&�=Р���=�ё=�~���K�=�]K����G���Ϙ�*W������]=��;�e�=@ވ����K����=��C=���x��<P͆<��ӻc!̽|]�<_�x����8�,ҭ=Rͷ��`�=�����=�;��]��f�����<��=$�h�GV=�O^�H��@x�;��,=���=J��=[U�=���� �9�����@��b!��}�fY=�E�=���=��<��>�T����D��hI���gf=X��<x�&�� y� =�����
�<��\�|�C��=/-�=h}�<�f���Q�<��Ž0��;�g�������;_����%=x�x��ɴ=��;��ٽpD��4�; ��:��̽���:X�M=N�ͽj�V�`3�kE\=�$� �[�R�=P+=Ⱦ=DQ�=ɹ= �����r�s��=���M�;=G����J��|���&���<!���=���<h�I�����d�>�d��׻��^�%�����)I=�qj�< =5߀�����Q�;=�V� �8�mQ�=�L�=��$��P��P���H�<#=$��� �-� �<����i��<�ּ`�R�Gڔ�@�����=�|�=�z{< ���j��+y<�N}=]ߥ=��;�|�H�><�O�<Ȥ���&2=�gL���|���w�=`�����(���?���3��3������ ;�P���1�M����N]=�=�����J	<��<z����@�h�m< }<��X����=��8<,>�<g�����=�g�<�:����;�c�6�h=�K=�Z< �<Pn�;ie=�$=�E���=�m��i�<�s/�`~�<X旼X���"<<8�<�Ka=��N<���@�;
�=���< �=�@�<4���C�==\-�<�z�<��x<�/}:P�;iO6=���<�x�=P�7<�~=��m���L=�V	���3��ӻ0Ԃ;@H�;`���/" =(�'��.�@A��dnF=�̫;�|7;f[P=��l;�{��<�=�9p=��'=@��<��"= ��8�=�[3=�te�;��=�����E�D�x۽�i�0�� ��<��<{�y=��U����Ľ }�<�=�������<�9&<P��;�>����<l�輨�{��$��A=tv����=v����<1�ϼ���XY=*©=�B)�7��=5F)=<���8�����_<�؎=�V�=B�'=8�ۼ�K�� �m�
ڼ��C;�[6�$n���¼";=/��=�vV�LM�=^ڜ�.꼔��>���7=���<��m�(�T��������ବ�b ;����8j=⧛=W�<hV���?G<�ۗ���P���@�R;�ļ�oW����<��D�z�[=|�<��|�X�}� �2��]���j�l���@�+<2���������xN�<h$��h�1<p黻@򁼈�C<g#=ȸ7<>�<L�����=��;���= C���[�謧�`�����ݻ��˽�.<=�e���B���j�?��=8\��5���~}������=�H�����<oK��fF����3=����X3<}�T=�<�5,���$�	��/3�d9�<�N�0�L��Ŷ;���IO���Ҽ(�3��ta�b�6�s=-�P=� ��@C���愽��8�<i�<tj=p5�;���������'�з���i<��%�_Y�W���3= ��9dH���3C��ڥ:H"�&��L�ԼQ)<�b�<��6.�˕"=p�l�@����6��L=�$㞼X�N��� <PN�;�[�r~�=Ծ�<���<�۱��/<8�f<6�t�h��<�	����<�ǆ<�r�@`�`r�;w#= �~; ���`Y[;�ץ���=.�� 5ں�Z�P0�� ^��0�Ȼ%zQ= ��9l9@�*̻Z9Q=��U<�0׻����n���	�<�$<<���<�I�: �չ47���6W<:B<�j�=0w-<@�6;@g�;	=��������绀�0�d޼��/����<���@�O��k;n�=H�<��ܼhS=0�<�0RV=�t6=�<h��σ;������J=��j<�������<�a<q}��諾$SR� b��Z��=*<��r��=|i=��	�f=5�@�S��V��O���z��۫<��t=	��= �:�zt��擼���<�r���Y= 5}���*=Pzu<�H�<�3�D5�Vp�=p�a���,�#a=nŒ=��A=V=�2�D��=�3���= ?���;g�;=������ܼ4��<����Z��l�=�o�= Ka�6�X��<��<p�N<�k�<�M�<��>��QM)=�&=�哼�-�=d,�� �ƺ��X<<L�<�P�<�pG�4�����μ ��:i�=�\�=��������=4'���$��#/=�<�F<4�ἈH���νp�p��c �@ ���K��r�<�XW=�f8������X��T8�h��<؃{�(���ʽ觞<�,B�XL$<pc�=��;4F����=$�<��"<`�=<u�<(JW��g�=��{=��;p�Լ(�M<Ф�;��=Ӏ4=̍C=�99��wX=�7A���<,\���N=�e$���ۻ O�;�Xz�^3��s"=@Vλh�Z<�����f?�f+�07� d�;2�5�(�ɼ�w���<�#ȼ���:������<и<���<�Mż� ��psP��'=���\�ؼj��0&���曼^D"=`��`*ü��C�0K�;�C�:��ټhI<��h���V���,r< ����P��J�W� %���.�в��0�;,#<.��x��=G�Β=r����<`�]<QB�\ꤼ@��PR�<@k; �M<���<d�뼈{�<`v�<������@nM�t��<�K���z�@�@*�:��<󃻤�<h��<�����< B��L��� f�������9G� �;���� I�;���:<9�왃<�p���y:�S�<Bu��<�ۼd۳� �.����:�jO:x<����b;p�0l���.=�f����2=`Լ��<�r`�J
����ż�����f�;�䡺�ٺ���<x\<�����S=V2=��g��.&=��;�wͼ��w��|�h�_�\ �<|�C�lQ�=�l=�2��������D< �+����� �<wĘ=��3=�X�8\���7s��9=�Q���l�=��Q���I=XaC<�a�<��!�`�;���=[�<0���HE�<$�<O=��=�Uq�e��= g��S�=8����ջ�!<f�����2�<��弘<�$�=��=�[ʻ��D���<|��<��;�<�"Ї=Vn>rS�����`�M<���3v�=P�Ѽ Q�:�g3<bT#=��=V������k��Is=`�<˪=�����Nc�ȉ=���� ��9F\�̗��؅p<l�<��^:�q����D%ּ��޼(;��餽K�?=(�<<�-» M���8����}=���<�������{5=^)D��@<�+?=�!=�ߔ��\�=�|�< �]��!l=��5=Ly^�A�O=��=0�8�|��<���<�H���-a=��-=�0�=,+��`�}; �&; W�;|g�N؀=���<|¼ ��P���U�1~v=��O��N]< ]a;־A�N�t�$3�� ��;���� �@��:(|�=��|�����`���A< HT<]=v��r 5�p}e��%=`u�� c4�^.�x�-�X�ۼ1�Y=�����*��E��XU#<؊��RdS�Ğ�`��;H�3�T���nhR��׼<�(λ����U��|���&�H<*� Ψ���;b�D���=8���DՏ<y����]I=X-=����p &��	l��̵<���< ��;h��<���AI*=y�=t�X�@�!; D�h�<����H~��(���±�'	=@�g�R�=�6?<L���|ƻ<`��;X�W���;xs�~w�xʷ<�΁�`D4;�)<,�ü@��;@��: �]:0�;=LU� "л�<��x�8<,㹼 q,:��&�1F<`��Ȑ���qE=��(��="����=�..�����><�<����̼�	�<�$�;��D;
=$)�<�^��Sh=��4= �ߺ�{'=�<���3�(�]J��_��l�6�H-l�=�L=�:���*�m�����< R�<�ݽr+&=��=���<���U������z�<�ʟ��S�=a��~�s=����߃:J�4��4��"�L=r}=8F�xo8=�Bٺn�=;�2=�?����=��<>�=<�
��6�������\�v$N��"<d�I�C�=g��=���=��λ W�<����H��<@م:��;��b�=���=�i?���t�p܋;D������=ߢ��Hݻ�V�<o�=�ej=��ڼ��;�o��K�= /;x� <�к��S�"�=����.�=v^�E`��PS5<�"d< ?C;�mJ�\
�L������z���ǽ�̈=M˼|����;=�9}<���s1�=�m=��� ȗ��h�=�]?�\��<p��<�2=8=¼((�=̷�<rWE����=,�=�M��������=4u8�p<1=@.�"�>��5c=�ξ<���=�����*��,�<�Ĕ�Ƚ�4��=p@T=8�ͼ�oG���D�����c�=t>���}<��b<���-����)ż�$ <%����:&�,I�<\�=p�������a8���/�<P��<^��=�&%�j�}�P����=�f;@O.<�?���Ѻ������=��إ)��O��@�D;8�����n�VG��ş:����ƚ���o���=��� wc�f� � X�8:z@�<⑼�����&�.�S�A> �k�@S�;=?���u=)�=��o���;�PT�C�0=<�!= �ڻ��<l�켍s|=N�c=�v��|�<��鼈-X<�����	�����x-W���=�k�; -e=�k<���<'=�/ =�HK;ؗ<�A
;jso���=�k-�`R&<�n�<��h<h<XQ<���=���h�F<���0��<����8�^�0(\�p<�<�(;�
���H=�4��˔<DM���F=�$��R��͜<�˹����2�=p��<蓈<�%0=��=t,ż�}=�G:=��M�W�Z=@I�6����D�f�J˫�������< N;\�=,z��%U���Ȁ�"�	=Ц�<���
=^�=�<�(ܽP�X���)����:(yż��=(@���ч=����Ǽ*W��I� 3S<(�=�Y�Uf`=��j� ��;���<Pis<㭄=�J\=��=H�d�Xu
�o�� ^��;0��Ő�\4�����<!��=��=���:f�=@=��䍃�`?��x���=��,=0�� ����4���Gu���=�,�������<s��=r�d=�:���D��޽�&�=t?��Hꄼ������w�1=�5����<=r�:�g�Խ��v���;�輼�V��U��P�S�>%��@T������uT==;伌d��_�=�F�<�"_<��_=��R=���v兽��=�ؘ����<p��;<��<��ռ��j;�=����@�=t.=�]�W����=�N�P)�<F�S���=L
�Ds�=��o�f�WmN=Їμ������=�|=앮��8I���o�-V��,�w=4�ټ�7��2�<�����ȽPj �H,<y=��8�2���U=�.[=�������;e��@�
=�$=/6�=���z�w���޺���<�!<�g�<:X����<���U��=Xd�x���T���B�X�� a��>�@��;ܲ��^� ��vE��&3=��8�Мﻌ���P,�;^�[��}�ݼH���Mg��s>�!�<�ሻ�Yҽ0]`=(��<�>6����;�����v=S�L=�v� �;�p��nę=�x�=�A]��i =dxϼ kG;'k��x
��Xe���햼�^�<���<�{�=P�;��1�5f0=�>=�2�<P�Q<`iT;�ŀ��= _���:�<TK�<����3<xv;<�Q�<K��=���l��<���)=���8v�� �@�<��< ��;d�V����<�����ȫ;x/Z<��b= 
ݻt2���N<@w$<j[�E�=
i=tL�< ^$= ;=�U��X�={)=�/�g�=��̼��ؒ{�r���Dh�����%=����w�<Ŀ鼕���5)Ͻ���<@5<Hh�� = ��<�<�:3���0i�lQ��>��T��e�=h��0�=����%:���������`x�;���=�����M= Lٹ8�y�@��:BO	=�JT=�vd=���=v�����R�50���H���޼���͹��g��~��=)��=�p�:���=��9�hZ���o�,����Y�=�q5�8}\�H���Dӊ��w��R1=8���&k�P#�<t�=�{4=���P��u���Z=`j���ƚ�0᧼I����=��(��o:=�=ּ�d��0��@�A��`��[��E׻���p�� /��t�O< �=��_:1�t= K	;0�{<��<J]=H7<X������=�����>=�Π����H��X�9���<k�۽�l==��\�=뛽���=�6�#��Z�>�����p�<�y���c=xؽ"���#ؔ=4�E�PҜ���=�>
=��ȼl���:R��ۋ��8#= '����ڼ��<�������r߼ �%<�?������[=4=����%�;J�a��b�<�=&��=h�����H@\���<�S�<HN�<��"�p��;ȁ���4=�zn��?ؼ�t�P������r�(��&����;��;��$����[�=�c �@���μ@@��>J9����L򋼈嗼��;�v�=�f=��:Ay���
/= i�<��;�@��;�0\� M"=��=nf&��a��R5�GkV=��=d�?p<�Ԣ���������0~޼L����i*� ��9��+i=������� �=@J�<�׍<��)<����q���o<���w�< V-9��:(�h_m<hE<!�=�o9<�Ė<�_�|b=���˼�6�`v�<��S���I� �; J�@"�; �<C=��;��8��_.<(�b<�ϼ�A�=��6=��<ч<�_�<���"Tv=�ˮ<pM���}=�ˊ�;��<ӽ ���P���x�2=��^< �=j����ݽ�<��L,�<@�T�d��<�Ã< %�;��H� �<��q��=T��΋�/,=�� ���=�p�����k��F����<�=�I���==DD�<�����}���Ŝ<[�U=���=,Ԓ=�۽ �c�H�	�"F-��$ �da��cֽ�\l�/+�=0*�=������=�2��T^���� ��*c<=�¼��B���.�����!�@Ao;�L����=ܼ�=Y25=��*��-���཰�<`"��������A%��l*�<\v��j�,=@#��`̽ �A�����\x�~&���������2�H���佈�D� ��;8��<`��;����c�:��{; �|<Ж=����w��=}���"eG=�f���t����t�>���<�M�V�'= CB����;���8�=�C�1H��ύ����.C��W� ��<L���:��l�=�M�������
=��(�R���P��d@�j|�4ӵ<�f����ϼ�҅;�F���q�l�� ��:���n�x\=�<>Qa��v��J�H� 񆻨"d<,VN=P������h��h���P��<@T<�f� ��9���p��< �;���8샼��a;����4	�B�����;�#����D��B��Z=�^�`�;���x�n�$B���9;�`���TƗ�T�Aa=��= =/�9����=�s�<��c��g$<�N�`�p< ��;Np�p/��&=��!= ��9�ڼ�@�5��z����ݻb�h�:G�����7�H�V�P���H7\=�5�����dc�<�-�<з�;�!;�F$�s2�����;0�ϼ 6y<�����;�f3�`>�� �ͷ/��=�y�<p5��_<p��<R�H���������F<z$�` � '>� {;�1�l��<��.=8L�<l�0�	<�t�<���GP]=�|= �-��]'�`�(�����2�=�*:�� ����h�ռ����,c�dw{��J=���=�z��%��=�T�=0ƻ�N���r�<��;�F�S���v�:=���=È�=�<e<�#�t$���?��0"&��e=�D�;[�"=�"�JJ-=���:��=4JJ��U;=�nZ<�>=�Y�=��=�ȶ��y!=V���fz<0y�;�2���=v_���K��G=�Ѽ�:ƽW!�=�ҝ=�3�;Խ�V��tHQ=�*j� 	�:���<Vk">�	���{<���<�㕽}5�=�Rl�`��;x�<�P<���=�B�����[���i=@l<gV�=h6��F�m����;ĊJ�d�ռ�O�*��=O�<#5S=��;�"������B�@�<,��/����u��=<��h;�t@�7/��1-�� *�<�#��@1E�ڿ���=QЁ�8�<�
�=2k� *w9�*�=4��<Զ=����<8���r��d��=�.A=�K����7<�I��Jc=��<rO=��i=F����==��u���$; ��Ȁ�<�r� !�9�{�<��B�>�M�r<X=�񆺠";�N�����ZO��[-���<0����P���%,<&�@���<6�.�>{=P1���=����,^�<U��t= ߾<���0U�����<�|��} =��k��q;P?�; �<�^R<(FW����Q\=Pe�� �0��A<x�=�D< <�9N�e� �ɻzm�D����>���+�������= ��< �{<*>1��A�<@�v<��=����;+.<؜=��<�}ȼp��B~#���F=*�=�%	���܏�<�%�<!Ӕ��������;$�<�=����y�=�������;A�i=@�_< Ղ9_62=T毼b�s��ގ;ĥ���r�< ��:L�ܼ�$�;����3g<��	=Ps�t����t��fC<|M�<������ռ��=�_�p�ƼX_<���(�.= ��<��< W�:^�Z��9����<����̧�<��<.<<�!$=4��<x�Y�>��=�h�<P8���Ƽt//��X���0P�`��n=b����%=�#�=0�ӻ�54���;�n�< u�9?��N�@=��=k�C=�_��54�`z��0R<�ʼX��=��'��J<�����=.���mT=R�=8ӛ�F�= *�;�ߑ�J/�=���<�yc�`M5<�(�����< �����@����q���!P�,T=�@ż0�߼�>M�q=5��Ƒb�pj<@=����ȵ��@P=iz�=:����t�����:����v>6������ �6�@̯�E��=���������#��=�k<b�L=8l,��6!���;Xt|����#o���r<h�v<s==H�!<Y�N��p&��0ʼت#<n(]���<��̼��L;���<ZM9�b�D�r_:=lt�<�p���l��x�=�都XL-<�(2=��=ؿt<:�=���<���<h"<L��<����E+=| =�-W���=`�;�/<(�V<̃�<-o{=��]O<�维�<�9м@˦:(Em�X>��P<dlʼ��a�{_= E�;P��;@�ͺa��8�������5� $\�T��@�*�K�=�(��c�<���Ѯ<��M�P��<�y��� T<H���$O�<�A=|��K��ڄ<�a���Z�<P;Ȼ�W��h�<���<Ŗ<H����J���=$����x���܅<4N�<��6< x}�b)N��\廾�%�<K��B��U��� � ��=R=��(;��)��\ =�x�<��6���<�<hN�<��<����8x��,�hY=�<����p`��$@�<Ҏ<�m��Ғ4���:<�ˤ<��=�߻p_= ����-�<3K8=�Iz<��u��cp=�!�r�R��u�;>��p�<0>�0����:�h���z><DT=��� �9�{�;��)<��< ��9l�	�� �<x�"�������;`���m�[=���< f
=��5<�F�(&���]8=�&��l�=p)<0=�;4I�<���<X��[Q�=��i<�h�;0�Y�`�I�d.弼����"%��A����<�Z;�F���v=g�������Ǽt �<`�!ĉ�H)D=�=��<ۘ��P�^�0��;�^<0O��!ͭ=LE������D�; ������Ȟb=�.{<P�:< $0;���<�G�hHh=��;pŽ��»�?��=p��3S���6vR���e�?	=��� �q��p�=�=����0℻���<|N�<�#���0�V�a=[+Q=�����F��$S��1<���=�;]������=��b/�PY=��Hб�1��{�= ��;x��<X��C�� <p��;�a�EG��t��`��;�h�<�*��d�<�
��C_��f,�a�B=/d����<����KĻp�=�4��xe���� =��=}�����4��<��W����;�Ȅ<�"�=L��<R<=\�<�|��8<�$=H��|#����><�zS����= QA���"���� X�9[D=pż�@��j<Py�<t��@�<��m��h�`�<x����m�_c?=PN�;P��;���� ���� �����: 0q��7�3<|��<�90�'
=�Լ�<@»\)�<@_d�`��;PVN�|��<��#=ب0��d����<@�x�<�x�@���0�'<4��<���<�}��=�S=c��x��<X��<H5#<�����@z���*������мЌ��D,ռ�b�=Yh9=��b�/�4!=(x<�f����<�'�<�Y=|x�<�O���y��|��*+a=���< p��PG<t�<�� <����RP"��rb<�K<�ư<@� ;�=u=@9���V[<g�7=�m�<`��;Tjo=(��(��'K;��`�<��9� ���i��h�L�$.�<�No=`e@; �< �~:'J<P=<ЙĻ@$�&�	=���Ȃo�P��P�»��c=5V=� =0݄<���ͼ�lU=H͐�W4Z=��<��;4��<��< �h*�=0�;@W?;���:\rG�����@�����;@����m����9:8��`y?�d���,�ܼ_6���ި<���旽_|=0;A<�R7������3e��|�; ĝ;4ɼ?�~=,ր�*8$�hsG<��B��������<؃#�l+�<���P`�<򼐸�<8ԥ��a<�Z���L����T=Po��������ʚp��-�@��:@#���$x���=��=�H��UK=x�<`-� ����J���b(=N�,��� <�x�0�ٻ��x=�̚=\>����u�p������(��<�S����[I��ȹ= DJ9 &ɺX
������Y<�M<��3����,���-���H�:������=�&_�4��x�^�?j=����X W�Ğ�����c:�=(���<�s�<L;�< ��ı8���{<�,���t;���2K=�F�<�i'�0��<R��L�:�t0=@�!��X{��o�l�1��6�=4P���{��F�t�\��V�<��x�����Ie=(�1<�^M���<8�D��17���#<��P�1���=�L���%�����k��zBh�JG��N�: Q�;@]$�,��<�)��T��jT=t젼B�==��<���<�7�� p�8 ~!�P�o<�=�~�;�Q��%6=�,L�`�<^"<���:��; Ay�,��<87��R�,����<P�%�0��J�=�\�<�ي;\򍼞R
� �����;��A������
�B�����=�8W=�мB�v�Ф�<�W��`RŻ���<t��<}[F=,1�<�霽�v� �i�ٟq=��<h�4����<4�< �/��������u<@�� t�9�p�;a%�=��Z;���9�X=@�<�.E<�%R=Vm��>�pn��27��ٻ<$���w���K~�(r���l�<�=�j�<h��<0'���a<�<H���d�8�= �%�(��`Y��D����o.=5Ė=ln=�-^<���� ���O]=�O�f�=�+=�G;xCH<���<`'_�W:�=���;�^��ȋS<d�K�Ȅ�<@=�; )*�4��<�gO���=*#��w�t0����o�����7�<����I���y�<@{b���Y������G� �� �_�l���/=42��ʼ@���wi��`�M�D뼔^k�z�=�n���<�D��'r��Hy�<Dwʼ�`��r��=.������Pr��:�t�������l���n=�Lc�<��{=��n� �=��Z;�틽������P�=SC̽dn�<�G}�����V�=(�/=@㟼&����՛����$��<�o��R7������=�b�(�a��Q��M$��X��<���0򼏞���޽��4�lf޼U�ý(�=�� <h��� Aw�V3=�¥�Ω����;��;9¨=��a�T+�<`�N�� �<��J�d/��	�<��½@/};�����jE< ��;�$�����<����p� �P�U= ����3Խ��0��'&���<JT8��ǒ��D��`׽���;K?ངo2�dR�=��Լ��5�8X< ݼ<6����_�ZSr���< �`�DO��@�z�@�!;j�� �8�@��;�<����O�<WH�ψ���H=@�����<�<P7^<��;��`<������
"=�X�<(f[�Db�<�q� �(�hJ<D�����|�@%�;�]�<�qԼ+��0
<�&<lU�@�<\��< -�`I��nڼ ��>��PJ+�0��޷9��/̼�=�r6=�����a�0]�<��E;p������; �}<0��<h�<a-���}�@����=���;���;�����;D�˼O؄��.L��JC<�.�J������d9= �߻P�M�$�=8v?����;'Y=f J��9T��i�2��o<\ʼ ��;(C���P�8�<��=(ھ<��;pK�;���<�৺�� �G����<$ �� �:�$�� %g����<� �=T�=�W�<Hc���4#=����}n=��<&�0˻���; -h;>{�= h���[��vI<p'd��B�<�k<�C�a�E=hԢ���=X.��~˻����>��T9��
�<jq��i��,��<��1<8�0��U� $.�����,c����ɛ<��7:��<�.$��˽ 9�ɍν�|��A=R0��@��;0��;8���,$�<��=�l~�<`N�=���� !y�Tt���+�:�`�R޽�fz� |i;Y��=�i�έ=���W��lf��f(����<��ӽ0k �`h�h�<�ը<���<ܞ�f8ԽH��<@[o�0��<�E'���)�ֽ�W�=��}���u������hi<�F\� .92�[����H�)�.�:�y�(-<�u8;zm��Aw��k�%N���UF��9J<�=6G;=�����#<���p�<�y1<�+]��w=S��8"-<�ʲ��� �������=�<#�ɽ�Vj�h��<���<$V�X�x<,fi���9�����٣���J[�u���ɻl���9g�n��=�}����ü@��;�Bܼ��B� �¼Ȕ����r����;�j"���� �x��C,<�X<��B�� ��; ͙��+��=��'�g���-<P�'��ȟ;H�M<Tɡ< �; �?��0���6�N0=�<�PǼ���<Pjռ��s�T~�<$��(n�� 9 ҹD���$���8����;2D?�P]<���<`�ջ0��;V��(�O��y� <Z�4o�>b>�,�����=��=Zj����7�=��<��@�; K���.+<��=�)/ϽL��`?����*=_���;䍌� xt��$��OW���H��8�X��*"�0X� ��K=H1��޹����<P�
� �9H��<>c����\�м��y�P�;�y���,<<0�2�����Hj�K?�=�7�< #���H	<h��<�s׼\������+�<,��8=�>��W�:��ܺ�h�=��+=P��<��Ǽ�8ռ@P=�ж�|nt==�<<ܦ�h�Ӽ�<r��P�<2�=����g��@����J����* �`�,��o�<F&�=� ����=59�=����IX<��H= � X�:��C��J�<���=R�=�:�<H7g�Px5��;L'�����<�J|;pӁ;�;��D\=�*���td=��=^���`5�= >z� ������=��=Tm���O��ս̞�P��<�帻��<@C�:v�y���=���<rҜ����=�b0<$�	�X���<�8�=ĭ���
�:h�����>����f�;���<�±�PH >�!ɽ�m��hu(��/��ֲ=f�P� �9:2=�v=���<ɇ�=$#���L�<�����4�dz>��O�y0�= a�;�ѐ=x�'=p	�<}y� C-�Ү< z<p[���|�𽈼����ߑ�ͽ���{�� fJ; �|�@�׻�ή����<f+c�ԡ���A= �@�(�<��>�=⮄=����c߼ē�w�)=��<®����=8�`�� �= ����p=hQ< ��;��n=H��J1=��o< �"��l���E3<�<�<�
/�E����]=p�󻀟�: w��1J�#y��@���Ђ���F���6E� �к��t�>*g�m�w=�hh��CK=p����y=�pټ��<@���+�<���<����Tb��v�=�v����=$�<$M�< <�6<�#=�/m;r�>�j?=����`���Q=��=(��<����ʬE� �a��8�P����b�������6����=$��=@F˼��D� ԅ<���ʴ!� ��<��s<<�$=D��<^�R��&�����=�4*=t�����<�=0��<���.6���<�ݣ<�9#= �;���=&]� �����t=T�6=�
�;���=>"�/刽�wA;
��t,= h޹R׼`�s<0���P=��O=���`c�;@��:d$�<�F�<����ɼ*s=��; �� ȧ��� �=��6=�7==���t�hB2����=��Pj=�u+=Wu<` =�>=���;:��=`&Z< bz�t�� %d� �:�?	�У�;�$2����=p��� \�8	��=p/��P�=�-�<`��;`p��X�M�\��<�Μ=�=�	.����X�;��<x��ҙG=@�.�Τ0����;�n=<���h*>0�<^ٍ�;�X=��U��y�s��=`�;�D3/�w�\�Խ\Ӕ����;���@���h��j�X�^ү=��<x�&�tC�=�9��
B���ӓ=�(�= ?x�pN�; ���*�=|]z�����a����[�>Ͻ�;!�2�Ih����u=b�2� {�;��%=�/i=z=��<=4#ڼ(`���K��|�;N����{��F�2=@A�6��=�=쌠=����<�>�,���ƥ�=V2$��^Ǽ���PTV���T=:mO�X����<�L�<�߼P{� s��J�?�֙M�X7@<��e=��`=��=�.=+In=ZJ5���<����/=�2���2�E�=�Z�<�F{=�)A�`=�F�;�<�=�a�H�=P���)��|μ�H<��)=č��}��p�C=�; ����/��R����ʼHz<�����E<�'0� �:�#�<V�H�}ք=N�3���.=@�B�|��<��s��@J<X�r�즜<�)�< r��I4:�v�<��8\�<��<D�<(��<�ݶ<�BV=@�㺤���W.V=(�i����;	�v=X�<�Vk<�-���k�E <Z�3��W� 1����e!�Յ�=���=�������;����d�߼��:=�3�<�%=��<}6��G#�X�V�-�=ϵ<�w�;�[�<�lU=,�<ׇ������!=���< �<�};<f�t=��L� �R�R�=)vk=�<�i�=�:�^B#� �z�j���9=亜� ��䩃<�_����=���=�:-<x�=�X-<`��<��<8���Ǽ&a=P�;`a��0o�������d=I�m=�7=`��:�E���Hl�= ,:�ph=\��<П<�e<�u=�A�<GV�= J�pM�<�Sb�X�ȍ <���xHi=�:���&�=<)\������W<�(b��(=�3��`�<��-��Ψ<���<$:�< FS�:���½�,=���<`��9&=r�����P�3= p�8����m&>�?4�<lL�,��<@��:���̋�=$|1�Ϲ;)Ҍ��4ʽ0���8@t��T�:dDȼ"�*�݇�=�J�<��M<���=�瞽���;�餼���=��<�΀��a4�@WڻRB�8�&<�ݎ�����=#��=�½��z��.F�%ٽ�a<����a&�4 �<�Y=���<P�{<`E��X̥��o��G`=dD����ѽ@]q;|��^+=n໼)>��ۼtz��|�Z�i�>�4g�Z��F��¼���=��)��ڼ�y�<���<��7�T��$�2�zI���G��&���[�=v �=X���+3=!"=!쐽Cir=()1�0֏;>Խ�����=�J=.�=qh�� ��:�.$���;�h< 4Z�9��=�ZT��դ;�և� =�<��B=��+�����er=��D�`�ǻ8����Ƽ��
���<�f��x,�<��9��@�< *�&vw��t�=&�4�=�w�;p�~< 㥻P� ��d:�<_�<��h<  L�n�= xM:9*=��J=O==BQ=�p����=�� �^�	���h=�7����<�g�=���<�,�;L�� n�H�<�b�(�b��z���:�vJM�rZ�=C�=�[Y�^����,)S� ��:jxD=b�?=��j=4:�<
x�L]����;�C�= 5=�ø<0#=�Sz=�ϱ<����h-���C=�X(;(U<D�=�l�= �P;L���� =̱�=�a<�:�=����V���t��ݼ�6C= uͼ��U���?=�x`��e=O��=8F�<�_D=@ע�l[�<��=��� 4Լo4T=(�]<�懼�����O�Ȩ~=$��=VG!=��ѻb�#�d)I�@h�=@̉<X�=�A=`|�;�M�;|Ȝ=m�=�9�= ˓�H��<r(��/�T�<�9.��־=���<X�\=`м<��N/(�d���pG�<Mt��  +���o��E�<���<����䚼�?P�P6Ža�$=@�<܇$��Հ<|dt�F?��	^=�0� ��r;>�ֽ4M)����'�;)%���I�<`ֈ��� =�I���\��@ 4<�.c�.I���� <jV�|z�d��<�. ��/����<򨚽 g�;(A�<��=�[���Z��,����b����`ym=Hyۼ�t��r�>ks=�_{�����6���K���ݼx�ؼ�38� �r:�|]=@�5<����+��S��9��;�=�� �Pѽr%�����y:��w�I�>H�<�q��Dw�;%>e�~
P��V�ʼ1f�=w�`<��;��;<�G� ���틽n�8�u �l$	��be=���= K޽*-=(Z�<sp��aW�=�1l���7��w�丹=(��<���;�s潸I8�辏�������;��<�6�=psp�\��<��'����<�B=�ڷ�����n�<|�ļP����1��4��o����<xln��5�<�Ur����<\T�������>,��L�=�:�<h�"<@�?;�m���:/;X�K<��B<��=<ټ�1n=����hj=b��=�=���<��*�`��=p&�X��MC=TD��Φ=���=�<�<�0Ի U� `<���<�����ۼ m⼆E{�����
��=���=mץ�.?o���ͼĒ��huf<�~
=߲A=[�=@=�n���Ə��#k<���=X�8=�E�<O�e=�Y=���;�N꽨)l<�=�nȼ�� �&s;=	�=��<(�����U=�՝=`؀<�գ=���W�������μ��1=X���T&�)�=��{2�=�<�=%=�4B=��ּ�r<�=�����g�JZW=8�U<�M��N'������AQ=k�=p =����ji-�4㕽*Z�=Ѐ�<���=�-i=@D=�`�"�h��=�t =�#�=�ˤ;P��;����=��/I=0��;>ͭ=��m=��;D<+�?ր��@�� H��Sbڽ B��n�s����$�<fh�h���lf�6v��Pe�<�֫:ЗF��������b=�t��<�D��:�<�X�=3����ݼп$������R��$��*c�N�	=a祽�v��p��<d^��PA��`,/;�L�dX̼ ����
E� (ؼ�N��'���ؒ;~�b=��=�:t�̒��lh��Vf��<�rZ�=(|6����/�>\F�<�e�6��L�������lN޼`�����}�&��Sy�=l��B��p�ϼ��ͽ &d:�f=����E����½l9Y�n3�_ǽg�=ܾ=����l�,^�=��L�`���;�8!�K{�=����L)�<�,����
<p�%��� ��pL�C{���ڼ���0[<�yh=h��F=����yY���1�=�Jy<A�ٽ\
�@��7�g=��Z�K �S,޽�۱�U��mך������=��=P�E����<䣎��aM;h��<PͽM���й�;d���D�'���*��T���u��~;@��:��;`�j����<�ּC�ڽ6��=��';�=�l%= 0P8�sr<L4��`��@�)�H�<2�8=�U��N�@=�\�d�<SEo=���;x�}<�B�ص=Y��Z�1�h��<P��;�A<�h]=~�<�W|�@������<��������Ǽq���Sʈ�k9�=�Y=5J����Y� ����Tz��X�< �<�Q=-w=�z�<C���>o���<�rj==��?=Xq�<@j�<�ku�ս����ʒ0=�q�T6����<U��=��P<Z7����(=���<8Y<x��=J"�6�V��R�����<>�Р.<��T=�Ỽ�b-=-.�=F�<Է�<��꼀��<< �<N������lX5=@DJ;Ц��p�V��q�e=���=��=(O��0������4�Z= H�<v �=)�=k��P��1:~=��=3�= ��: _~�ص��&e��v=�h�<C0=2��=�6��`D=�,ֽ�3(�~g��V�r���� <hH��3F����< 嶻����t��[H��H<�8��TzE���L�4�<4��� ��SGƽ`9�<�n���ٽ@d���~��d���6��eڼ����<����м7>=n����̽����F���������[����!��0�����<Й�;�_=@}<R��`U
��(�� ���,� �&=�]���<;jS�=��<@�¼�Sѽ��$;[ý��g��k���[N������=L���pV�p��������;�9�;0D�te��>���"s���]��mƽ�Ղ=��=B���,�;�J�=��A�Zi�P�;�Ӄ<.��=`L����<�鍼쫐<�`y��O�� }�:w�˽�ꃼа������`�<��Ľ4�<�V��>>���a=\�.=?"� ��01C��<�Y�
�f�ӛ�� ��,&���A����ݼ�A�=$t���¼x��<`��̨��h'u��$��I��� ��9*m�p8p�$6x�XE(��ֶ�`p����A<��� ؔ�[z8=ԡݼ�
Ϧ= �d�'hu=��X=��?<���Nb�.D��S'���<��[=ZGI��[=�j�П�<�nn=<���P���`>)�$��<N*��KU��< ,���]��sE =�\!=��R� H��@��Ȭ<{�dD�t�*�|���^�����=\�?=[Cǽ������<D�A�xP<����
m<8\=PT���#�f�\���M;���=�[=l!$=���<�A�P�>�䙼`�H<P�1��C��@`:�~�=��*��ė���%=�}�<�X<:L=n�%��췽8S_�F	x�d��<ޤ��iI<<
�<�Y%��<��=3�<�{�;X��R�=@��ǽ����cR=�F�LB��h�Q���e��H����=�G= �O��
%�0ˀ�+rA=@�n;���=s!o=>����-d&=$5�<@��= �; ��.@��	j��4d<<YJ�H�J�pgW��M>�ص���q=�=�=<ݼ�S=#P�=hwS���$� ��ڡ�?y=�7=�,=�-����V����<p������C���T�в�;��l=�A��,��=]98=�ս�Aw=2��Cq����=��<�=���Qٽ��`4v<� ;Ȑ<�Qu<*Ml��8�=�i�=��꼰b�<�b� ���"p ���=t��=��ܼP�Z<�nN��=�=������];PC�<n���o�=S�?��R/��T��G�=�qj�he=�\�=��;�K=LU]=P<����im�(o��k���'-���=��`m=V=�I�=���Ą)�\�<���=D<W���Z��S˼ KR�ƫ}�5ó������/�<P�Z���0��+�;�3�#�˽x[<t=9�0=9P�=�n�<)��=6Yd�8�y�:���Dr�<駼(�C�(�<��<���=r3��u�=�4�(Ol=�w=(>>�؂�= �:�V�����:� )<P� =�(�����R�1=pތ�@`<@�: �A9-4�`:5���� <������O��;\����r=�:��|��<�"���<0�ؼ�廈�s� ��9�_�;����E��`U�<@9-�P�< ��<4C�<P<`8V;]�&=�Ç<|�ż��#=��`|;
Y)= *�;��<�� :�t�@'o; 4�������R�|�������֯=]�d=^�B�6�/���; څ�N����< t��Զ�<h�#<����\���c�Yd�=,�<p�W���=��=pd�<�G��أټ��z;@�R<@E7=p�b<��S=xP��@>^����<�l�=��Q�d��=J���8����<:���|=�;��p�i<�P1�`g�<�O=�%3�\��<��<@%E<�k�;@/
���W�ȿ}=d��< �ٺ@�f;p�ܼ ��; M<�;V=葉���N��1����d=��	�})=9�=��<Pԇ<��=�N�<d1�=��I<\H�<��ѽ��[����< �B�]�=y�4� >v���������F=̹̼��^=��b=0�b�*�?��zL=TM���<0R<�b+<|,ʽH�|<�5#=X�3�8S���~�u��ɪT=+'=�����E/>hvW��^۽�\=�]���#��=�������c��.��B�p� :;|��� z�;(�f<��"�5�=���=��<�G =����l<�=n>�=8�a����<d�r�RK
= ��º��<��,; �=)�8R[�ǈ��[��d�<�r����<�=tԼ��o=D�<�r�\��z�b�H��<�9�Ew���J�=@Aм���=Շ;=�">i���p�@�P�ݼ��>�Ʊ���,�r�&�ήI�u�'=2X����P�Ż0��<��Ƽܷ��tBQ�l��)����N��m�=��=Sj=��=��=98��$��<�}�`)=(����
�~\�=t�= d�=�*���ˋ=Zch��ʒ=�20=��:���>|���M������V<�X$=�8�<H���=4ؗ<��$<��p<�I���� <��<ގ�pW�<��9����Dp=�"���=V�\� �7<XG[��z5<ܹ�� v:�뀼`�n���s;p����	<���;@�8<p/<80�<�<��=���<��W=p�l< ��c3=�w��(�8<��7=��O� 
�< Qp:�a��0)�<�l�;���<�Ϡ<����<��b�d=�o=�x]��OD�P����έ����&2/=HIq<@��:��R������Ҽ &��yԊ=�
�:@�o<�U�<��F=���<�������F=�<�=��<ȧ�<`�� ����(w:x��= �W9��=�g��n�T��<�a޼��;=�27��Տ����<PE�����<��\=��j<?A!=L��<��C< �;�ݕ<@N�o�d=���<���<@t�0�h�8�<��<T�;=@$��D ̼`)��lQ�=�Au<8v�<@��;��'<P�λa�M=~eP=�f0= ���WE7=�a��\"C�X��<��F����=,-%�uz>�����!�81p�
-�?��=�<�?i�[T��N�=���p����ּ.6���2�y=|=D���TT�j%ڽ�	=�E��=0��;X�(�yz>�����ȽJu=1�����)�=�1����A=��ٽ)�	���\��������;P��;����|�=.��=�"�<���<�<8��<�q%���/>�\=�3b�x<�`���Ƚ��L=h<�@-�P�=^�=j�:Y��2Ͻ��5�����G� ����J�=01�7�=p=��\M��+ռ��0�;��=04�Pν(!�<,�
�Ģ8=@3�:��U>z��Xm]�f���B�9>��\�0�����Q͋��=. �.�1�`�;L�<��1� �
:<_�Z��ܽ����=���=�A��X�O=/S�=$���ɢ= _��P=��@�X��<䩿=��=��=���D�<���� I= �<@���J>L�Z����X»8��<kl_=`�������!�< .�;��� ��8PWU� ������<,���Dm�<�wm�r�<hq<<-˼���=�V�c&=�-<��<�0��V �p,� ����2;\4�<PK�;��<�K�<�!�<��K=i�=)=�I:>u�=(yd<�7��a= ���~�= �y=�<8a<�B����<,�!=x\�H]�<Dr�<,(����=Z�==
���4��4�����B��x�@,E=W=���<@?";�ˇ�Z'0��x�<�h�=�<�4=�P=l�r=H��<J�E���λG�%=���; ��<�X=��= =���f'��[<���=��<���=� ݺ�@�:<�~��:.d=t@���<�M=0���>=@ې=ؔ�<:c=��J<���<l޾<8���Z��|m=6@=X�<h���4�$�@u�<{�l=Q�,=�lH�Tc��􂄼U*�=D��<��u=|<�<ظ!<x��E�=�=[�J= k;��;=<Rf����O	=T�"��>�7���F�=�#��b�e���{���)��l~=������K����~>�.��w��81�Up��?�$�=�O�<�R��5���ǽ>Eg��Q�=lQȼ�J.< 2v>r'�����ṕ�L༴�Kf=~桽.�r=_G߽����e$�Ԍμ"*�����;8z ������e=�y=Ps�;�����*�@�K;z�;0�> ����������n���E�μ�=@s�����@M<>dK�</-��|�p�/F����H�I������d�Q��79=�@��x�/�.�X�)�~
:���鼓�>�^3�C!Ƚ"��h�)�`9�;PD���=>��;��J�ֶ��CQ/>�8a�Ł��H�A�G��r&>|��0活 �;�)˺��l�@�G<�C��v�3��'����G�b4(=�)>8h/�-�P=��=��%�;P�= ����*�;b�y��>D<k�=&��=�h=
!�8b���bd�p��;��<���<3<>�_�+�< 7ջ]�!=MZ=�t�k��t�<@�������|���߼�<���<0V��pE�;І���=��g�S�����=��T��c�=Ϟ=HZm<�FԻ�$� w��mŻ`p�L�=��֩F= �+<�S=�5�=�%+=k9=�9�rڍ=  �n��pR=<ڣ�h�6=��=�ߝ<`Nr�t'��y<<V"=V�+�@��;pG����!�U䆽t��=i��=cٻ�R0��Y�#����� =݁&=�Mx=�c�; �?5���
�<cȯ=��3=^�=Wq=�U=P{�<������<�J=�T��`��;V^4=�t�=x�,<Z`���g�<D��=`2F<[�=(W���ἰ��@s����`=�ۍ����;bҘ=P/��	��=d�=��
=)U=��(��R�<@.�< b��n����=�^�<�Ӽ����t���&�<x��=��2=<� �V��7>��q=x�
=/"�=�*]=��;��&�n�=h�p=��=p��;��=��e�\1��::=8�b���>�9=0�U=��?��oX�����Nu+��=�g}�(�k��+�����=p����Rf����#���������=��<�lFw�����X�C�,�M���=&�e�|�<�*)>:B��ߌ���5�D����v���D�={.=�!���rݽ��:<�|&��ɽ��������.����<P��;��%��5��̱Ͻ$�����<�޽=�E?����,͓�(O<��}W����=�5>��)����=>�W����C�,�{����tE1�/	���]q������ef�=Tr���@� �޼퇧������=4��%�������tbG��j��'G���7�=�x@=�Y:�V#��zQ�=F�:�|�����8]���> <���<��;�����Pm� t�;tR���cY�6�!��5A�0А����=��2�gvG=-
��n�����=�l<���^��Ц����==�D�f���FR�v
�����sI<�'�=��=8!�|��<8�m�ؔ�<���<X��z�z��w�;0S��*����.�@o�� zX< ��; �ٷ���=�3��m���c�=����,'z=��U=P��;�_Z; (]� �	����0�s<�d!=!���A=��*<���<Ö=��T<<ؒ<h����r=�sf����li�<��;t&�<�k=���<]� ���4:<��<&yL� �ܺh3*�>�n���]��$|=r=	�ƽ����ͻ��e���'<��u<�=46= ~ѻBqj�z�;�4�<W|=9�7=��v=�~�<��<�Ռ;u���@R�:(�C=hw���Қ�䙗<��=�c�;X�x��*�<r2=���<l�=���C�`�J�b�"���:=@�f�Qo=%.q=(�~�ę=�@�=���<�y�< ��� �<���<&g�����~�z=�A�<��R��xO��:g��ё<4��=O,.=�7ʼd�ټ pG�N�+=H��<�ÿ=��=����@U^�=�= �5=�ԡ=�@�;l��<�i=���L�H�Z=P1<��=DÆ=���;����{(���l�h>�p�\<]�ʽ 1�:pN�J=8�:<d�Ƽԟ7�qҀ���N��y=�����`�������;����R�I>���v7=S�=7�
����d戽�P���������(�ۼt� =����6��� .�<r����1Žp���V�8��Лa�8S	���伿aŽ��+�� *�c�<��5=T�~�(z-��ؽ�8	�fA?��Q�=4�e�\O/���>�x%��B̼`����p�� x��>����et�:���2�=nq�$�B� _���@�;�r=�^��:4������B���P�����V�=y�a=h�@���o�'�k=�V��]���)�L����=�P�<(�< vF�@�(<����YE�Ģ��,�������� �Hc+�|��=���>F!=Fo>��2����=L��<�v���������*=(1�����uƽ�m��|����� ��`γ=dy�<��T�'�<|�`��; g��#Ž�В��ݻ������&j��Љݻ�������X}�<�5��m]�Ad= ����G��J�= 5��7o=b�=�:@<��6�Pym���8��c�H�<M==��+�WZk=�����'�<�m�=�g�@5���ټ���<HKڼ"��84\< �)�h�$���5=H�;=�3���햼�PI;��<.2`�lEļ4�7j��"Vy���=�h=��vm���<��6��D<PE��୭<�g%=HJǼAE����0��?
<Tm�=��U=na=p�.<��컰� �=��0p&���<��-���� 0���ķ=�#�(���= ��<��<��U=�XּI9��@Ո������=@-����*=��=����i�<�P
>�/�<�t�;��ܼ��=�l&<L�ν���rs�= έ:�߳��+e�Hsl�h�d�L�>� O=�㼒m+�(�Y��=��A<�@�=�j=���� d���=x�<To�=P��;�?D�:���8w��p��<X�����o<|���>����=b�F=�
� [�<B��=���#���=Jn� ��9`��;2`J=�3���r���#=��/;��L����
^N�^�A=h!<=��8��=�,�<�H����=^*�;Y���J=p=�<����OO�,x���������*�;�\�<L��<6<����=���= <��Ǽ�$����<�l���=���=�������:��^�1O�=���K����(=r���\S=��t������۽�H7=�Ǝ��n=B�=�j��Af=�ٵ<J=$=���������#�X���f��X��=<��D��<���<�l�=����:�p>�;{�g=��;x㓼�,��Mм�#�5|�������S���=ܚ��`;�8Es��b��8 �� ����=R�=⩁=�~�D�h=�~��`TL;��L�,�=T�ɼ�i� (h��h�=�j�=*IK����=�ס��|�=K�c=2ҋ���=X�޼`b�O=�x<���<�7��X��tb=`_�5�<�@<\S�<j��𻎻p�Ҽ0x���2R<�j�<(�q<�t�;K���}�;�P<<�=�ռ:�@�޻@�o� #`�@��:�f��X��<�傻��<`	�< �l<�Z�;)C�0��<��<@Sٺ�w�<���8<�8�<p/һ.A= AW<��9�H|b<\*=��; _ܻK<�F����=��=.8I�FY�`�;@T%������<����Tt�<@Tͺ;�����S���f�=|��<H�T�p=X=g<�N�<Ri2�����������$W=D�<J�=H�J���ļ@6d;iE�=X�c���=h[Q��Ѽ
M�=�����<�Ϡ<��ϼd:�<�c���=�B�<@��:�ׁ<`��; /<�e���o;�-�;�hv=�D,=�<�O�<(m�HKݼ��ۼ��{=����Jż�5<��=�Ä��q=	5=��< k����<���<�= ��<	!=,���вy�D��<�����=�B��c02>td=�l�E�XlD<��*���7=<x�=��/�8���^�=WӍ�fA�����|��<��н�\��G=``=;^������ܽR��=�`;=�*��V$> �׽�;=J�E�#��0σ=$ ��@.�<�����b���͠�X�W�(�p<�=�sE��/�=��>��=�׏��� ��E�<�=�*�/>fN�=��k�t#�<���l<�!1��*� �<ؤz<�@=m��T�$��d���0#����6"=�+>Z_Խ]�r=�m�@��<]��������<X���g�0�=����e=�[�<�?9>�@ҽ���-���=������D�����y��i6<B�v�Q1��t���䮼<0���K<�ߤ�Vc��K���ɼ%�=xC�= ��:@�;&~�=����	=�,J�ˊ=�Խp:�<��=n��=x,�=U���8�=s�սL;�=ܦ6=fݖ���0>�;T��hż��<Ж�<���<���< 	H<�b�<���<��z<�P�<Lj�<��< ��<&��0�q<@p<�U[<a==��<�ۿ;
�t������=/<l��< ͩ���@�O�Xs����$��(,<8y<��;ء�<�)<���<�"�<��<���;2L)=��<�+�<(
=��w�0"7<��<�e�<,=�J�<t��<��<�&+=���<�J�<8�< ��:��C=584=.�I�L�����q� x�8��s�= ą��b^��%�xt�<ؠ� iN;��=@�s;�jA<��C=��<D/�<�4�@�G�0��@i<�s+=���<�u<<�䡼X^�������=(�'��%`=8J�`0<l��=�0O�E%6=`�< �;:l��< ��7�
=,p=���<.I=(p�<���<0��0%�<@<v<�g=W5=m�9=P�<`��;h2���C���m= �8;�mT��G�<�FC=���<$��<�7�<`9�<Pq|����<gzS=�`�<|`�<�ER=��x�ȨE��c�<����L�>(̽��3>p�l���O�Z
B�^�6A�=��<=N��q\���,,>�'j��^ͽ@�,�aX��齁�Q=(=缨����A��M��		>�`�<��O�]��>`��o�� ��<�XY�t�.�dr=ܴ����=������~�=�4Kļ�[T�0٣��9=��<_&�=�>��5=تq��g��ex<�'�$�C>���<���P�W<�Ͷ�y����=�x���Sc���>�N<������5��\&�x\�_䏽z�n� ��� f�=�\��3�<`�f�t�������b~�$�
>5\R�QϜ���<��(O=��@�2�h>bZN����㽦6>�y������>��AϽ8��=�����Z<x5<$?Y���<��֨R�����lb�	�=9.>����~�<�W�=V�@���=I�����=F�Z�t�=�Y^=#�'>�f�=
e�`��<�7��7�=0r�<<�S�D8�>���P|	�� <�=>}%=E�;����T)�<0��;P�*���<,��<�i1����<���Ȇ�<O�; ��<�=@V#��N@=Z���<���<(��<lX��@�ûA#��&���,�$��<��;L��<h��<��<},=Z=$�=�����=�ȶ<@^;T�K=,���h��<;�g=��<���<P٤��U�<0}.=L��<�#=�v�< u�:X�n��h=$��=�z�J�\Hͼ$��@x=�65B=@Q<��<`� @;��ۼ���<�"�=b�<0��</�p=�v+=�=�<j(�@�:��;@�];��<��=s�=�����`r�F.�= T��=`�����<�M=0�R��z=`ge;8X<tMS=�ש��|T=ډ=��%=��;=X'�<�*�<@�R<��u��ʦ;Y�=�5=4k�<��C<��xNI���=_5|= ���5?�@Q<�~l=/�<	rh=^C3=���<�����<g=�m=�)/=�<�fK=PI#�|M�hg�<t�f���*>�����p>�꒽����휽c�O�k=�s�<�:��������@>Q��BܽL)��Ƚ�ӽ3W�= TS<�_[����k���q�%Q�=�3&<�o<���>����蜽hx����/�` 6�Tl�<(���@�=�A���h��\���g�������j�<$�=@�=���=��<�y��T��Ɍ< ks;�Q>�������E�����uB���>|8�L ڽ!G>$ｼ#7��0༴G�X\�'Ž�5�L�T�B��=_e��t,��]��� ��/�����5>k�A�g\������Ȕ�8��<JGE�7>��*<��T�����">卆�я��pǸ�ycؽ� > �; |n;��=(�%�02��Ȭ�<�#��`m��$��B�����}=D<>��_��]=<��<�AD�Z�>H����]=�����9=�u=]f>�=:� �@W�E���==�e�< �P��Pt>����e8�,ρ�G�&=s==<<��l����s<��������¼���;T���h��<x�ټh�<<`f��]�= �b<�[-�\y�=��{�٫7=�h=Xĺ<`�� ���k!� � �߹8��<@�8���==�u�<�O�< �_=f�=̰�<��3�:��=�T<����&pE=�v��u�<��=�,m<@��<��żP��<#�"=��m�h~�<P��;����|Aܼ�=f�=�=����x��|<s�`�$���E=$��<8�=�[[�g��Ξ?�А�<��=�p�<�W=�G=�36=���<6xs���>;`ʛ<����`�<�)�<�R= 4>�d�V���<<���= ���Ϟ=(W_� v#��|<TF��=zl=��T�(��<��l=�$x��]=�w�=˚'=� =`V�;�^�<��<�o��"�=\s=�x�:�3�X�� {;V��=�Mt=L���l���P�V�Wq=d��<v�=0�O=�+<�H�����=��G=6�=P<<�=�3�@l0�T4�<�>���W>���ڶ�=�7��lmx�������=���= �;��"�*1V�Ј>�Զ�����\B�/-�ރ���ż=������m���(� jĽ<{X��%N=��=�l{�<�{D>��t�e��|s����@��������ς=�᜽`���?�;�R�:P����%� ��pm�<�u�<�k=��D;�׀�ս���$<� A<��=|I�4���	��P�W���?����=��x�*����=>.�F�]��W�ߗ��|0��ٖ��y�@^���f<�=Wӽz����;��ف�0�»��>
���f:]�i���,s���+�����Զ=�}l=pf�;�
ؽ�2�=2.�N!1��+��Y���V�>��<l7�<�=@���6ꅽ<a�<
��~iX�D�`qR� ��c�!>f$a��HB=�9�����=$�������Ud����O�R=���=�&�����@ؼ�f��9���ݝ<s$C=�>p�/�P���ȅ�F�<P��<ଲ��߼�; ��:�N��@Q�@Y�<�!k<`g�<��%<@�Z<�"�;Gr#=`��<���V�9=x���<�.+=x�<�\X�F)=8�?���>����<�< :\<lo/=�(�<@_�̅l=��<xDh<�J�<~)=P	<�P�;���<ܻ�<p�;kz=���<��; ���D�<���<�����<䫻<�开_N�L��<l'�=����Hi��=,��%μ ����=RP= �������׼�ռ�9�<�ex=���<B�f=Z�;�e�<�H�<�V!���Ҽ�u<=pM�; ��;@(R�q-G= T>��Jʻ�^<"�&=�/<�s�=����/ּL9�6�h�a�7= _k:}~�=|��<@߹�XAV<�L�=��=���<��A<ha�<��
=`�)�,|���6�=�OJ<�XB������xV�8k
<�H�=vQ]=�dֻ茵��dz��>=��<0V=��<H4�H`I�n M= �
=��=��ѻ%=y%�@�U����<-�;��=��c<���<xz*���M�ov��9L���<�>U���c:Q�t��=@��V�l��CI�3Q۽��N�pH�=X<-�X�H��d��|6� i%�8�J<���	)=`s�=D���8��j蝽�����Ǻ� �G�yO5=z{�D����J�;L-��`���~�S�$]ڼ@��:�2[;�\g�P켼)S�� �����@;�K<�~@=Bm���e?����� _7�b%�3��=���eǽ�z>Y�Τ�� %�C7���B�H�����7��{e�L��={ӽdᒽШ
��멽��6<H��=����4��-�ѽ(�b��K�S㺽�=G��=  8fʢ��)\=� ��f���%ͼ�̔��� > �	=���<d��<�G ��,���;�����L���H�'�L���(�=6/��� =�x�S*�����=��������L �ֱ�Z	= 7;�U��K���>P�K���"���<H�=�X=D���0�ü���� �;@�ɻ���Z���q�p���h��8�g�\3�< ��9Xz<FJ=x
� ]��fK=���;-�� V�<'4<$Z�<��R= �/;��i��a-=�욼䓽��=��R<@���%+=8<x)g�EPP=�.I� r`���<`R< :o:`�;`��;T��<x���D1=�&= �����;�s<�Jv<<ԼPz�;ܼ����>���K�<��z=��|���%e<		� >�9dҳ<�<0����`���Q��[���Qi;�Z=T�<��\=�����
����;:i�ں���'= ��H�� ���,v=H�c���
���<HsP<P�<�
9=:׼^T��{��C�����<���;�|�=pP�; su�h�
��q�=P`	=��û�8�:�G=,v�<��z�D�׼C%�=�0���J"��:��.� P9��>'*W=��<�t�꼔䨼,Z=ؾ/<�Q=̚�<\ʼ`W�"�=ȿ#<C�=�����(�<*
dtype0*'
_output_shapes
:�
�
Const_2Const*��
value��B���"��-�=4��<�k =f��� _�;��U�,_�=��ټܳc=!<��6;��C��(�*�z��%��V���
�X�K=p-<���H�(��=x$�$ڕ<����N��h�= ���V� ��;\d=п���ü���<Rd=��<ܑ�<�S=�*�=�Y���=�d�� y�;ˎ=�M��F�;���<�+<�U[���;���<N�&=4���sv�=��<̛�<�9;=�nM;��=2Ň�%�<=X
e=�S=�b=���H��Ƞ��p��t��]����m�:��;�S��ʉ=�`E=������T�?=`�&;��o�s`���=3= :#< �7:��1��t=�ݶ�t�O��м�N=�h�T��<���=�J�1ߐ��|�$<��`�;�Or��ꕽh1M��һ
�V���t=�J=���:�����T=�'�<w�� �=�9ļK��=���<.s=���϶�=���;�֟<{��=�&�兼��=�Kz��=4m��d5=�=.;��%� 7���m]���R��$c<p��l��<@�"���]�0�3�0���з����X���"�8��< ��9���� 4�� �~;@��: ��� $����߼@�:�=ȓ�����LK���x��d�����<(���w����� �����=�
8� ����V��01<��5�R�N�h*2� �g�h=��w!���q��3��@�;:�<<ߐ<<G��U=Z�w���=�Ug�x@�<�:�:�1Ҽ��������;8I�(��<��&=t�������s���c��TT�`�����<��A�L��< A���r�p,x<��X�Z��`= ��n%��ű�D��d>μ�a�.���9\�����X���@��� ��M<0��;�mn�@���v����)�l���I����;x�<H�z���� �ǻ`�����e= :]���F=(J���ջ@[� �`�Pʄ;�0Q��<tb���@az�@���z�; ���@�>��"=@ڴ�g�n=X��<���<�@����9;�tֽ ��<0�'<� l=8c =�*�����P�K����,��.�,�μD_�<��=����y�@θ�%�=P���c�%=��Q�Pe3=p�A=�<r� �$g�<�Y�<PW�<�d�� �r<�fʺțM<��={�Z=��=$�<�V~=���<���<�E�����<�nJ<�+�=4�P=�˄=�!:=�m�<C4N=�=�Gp<��='A0=4��=�|��5�Ư =���<�֥=����� Bu<<��<�X��9��ʶ<�l���<��M=(�<Xfk<f�x�4"2=h:p< �d���� Ѽ �ڻP�v<x�v<�u�<��u������-�X\l<��ѽ#;�=�]3=L��p�H�r�=�����"R=���<�{�(x����<�]ż�������<�=x���%�<v��=�'{���I=��=���.�E=�E�=L=P�ệ�x=�9����V=���=L�<����(�)<��9���`=0O�of�=K�H=@d׼��B�H�R����? =�PW�n/
=���;:q��;�,%��H�'�~�w���x�u�X�=��<��C�v�/��jɻ��<�q1=ļ��2�� �G��!=<�#� ��;R�7�^L0��u漧2d=|�'���i�ЭR� /��ѼG����q�l˼$�#����u	��p��;��^����r���������)�: y<d�<)��T�=�0���:=n�J�/==���<B�Q��b�`Z����O<�fP<P=l�h=D$����2<hv<T_f��E�;XcF���<����;p9������<��i��k.: 8�<�m�@�<� �y��d����Ҽ@!�;!/�P��<�O<�����H=<� ��K�;p�< -��v�=����`��+<��k<,���I�<V���h��`j�;�j��ܩ=P׵��<���@��<𳋼��?��j=$4Q� 7z���;(+����4n�<�ܥ<�� Y�XRc=��)�͊i=�W8<�*X�xP���j����E�'=�c=D#t= �:r)/����� ��<@q<u)���v���=���<'I���#����S�b= �ݼ䮑=́���=����ȇ<��
�8�L����<ʝ�=��b� =���P����H=�L=���=���="�=h9��Ј<�񭺐Ď�p�̼ i����S
>�)�=eC�=�z2=��=@�:#h)= �:�傻��=�o�=��5�l��� 6B;�UC�r��=>g)��o	��k6=̐=
�<ؼ��:=wؚ���w= )�; ����;=:v0��+\=�h<�'�=_v�_�������7�<�,=,+�(�q�U�<Ċ\�06A�� ���=0�e��f��G;'=Y�=���<ׯ�=��n=��Y�����*�=�J���tպ�n�;�"�<�|C���=fc=��R��+�=82]=L��x�Gh�=��߼��; �B<n�I��S�=)uL=:�V=@�2� �%��i<X�<���t>o5�=dB��.�� 3M��qr��o=lO�R�=�H�<��n����0?׼� 9��%���=����<���=�q"=��"����� _;�6e=�+�=J����̽��h<��U=��/��=o@���
�x�b�=<Y���A�8Zr�@����.�����<@ؼ�������J.�mA�B/=�_#���w�8�G�?�<��X� �y�(I<�U�<��l��>�5��=�K����y=�=�oe�  _�l6Ӽ=�M=7�\=`��<�6i=�k¼�DY=��d=�A����.=��W�h�=Jsc�x 
<�
� 
V�l��<ϣ<;�=P��<�%`�p>�;��7=�Cr;�-���p=��D�G��=�Zh<�Wڻ���<����T`�<�!=Tؕ<X��=��S�"9=��q�6n=f�5� �6��/��� ��<�Ἦ�=���� `F�l����2=��ü`λ�"Z=R ���
�hm�=�o=� =/�1=ҙU=P�`���r<�M�=���� �=���d?������dvK�ơ������=���<{��=����^�~8o�{n&=P�+=��彐u�;$C�<`��C� ��;>!��4<L�2�vE�=�.��ft�=
��� l<��ҼJ`:���g<7��=H�O�n==²1�����J�<�=��=��>i��=�:��8�n<�����	��dⰼ����(@���>�	�=���=��=Q��=Ԕ��U<<ඊ��jQ��p�=�3=�T�;�V��()��D6��[��=fQ[�TxJ��@'=��=�l=�?����(=�ʽF�E=c�\tb�P�=�e��&^=�;ә�=�����|GI��F�<��<~���1W< �J=���L(���Z@� ��=:jM�|�����=*֕=�u%=���=%[k=�M�,�*�S�=��P�0!�<�h��G#<@ST�>뻕�A=Ǹɽ���=G�=��*d�"
>V+��?Ҽռ��Ž�q=H��wFE=l��0Z��>�=Э�������>NX�=X�K��:���]��H��S�h=�$��<l��<�]�7�׽t�����dĽ0�R���=��=�=p����������<�A�=���=t׼�dϽ ��<{�V=�h�i�b=ij��q5���(�X��=���������B���������3K���`1��b0��a۽��;�н�r=h���&� �q;T�<Vs��;���\�;��I<Gӊ�tF">@��2�<k@ս�\=�=��<��[�,"��x�=���=�l<��=0|d��6�=(�=넽�|=O=�?�<�V�����< �	��ns��D�<��=���=���<�i���E�<6'�=���<`�����3=�WR�h/�=���<�$]<���< ���t��<��6=?�=m��=������K=��S��6^=~�1�0X��X0��0K<8 �<�3�糓=�̧�x�r�\����g=�����K9�rs]=�h��f�3� >�Z�=[L=yoD=P&�=Р���=�ё=�~���K�=�]K����G���Ϙ�*W������]=��;�e�=@ވ����K����=��C=���x��<P͆<��ӻc!̽|]�<_�x����8�,ҭ=Rͷ��`�=�����=�;��]��f�����<��=$�h�GV=�O^�H��@x�;��,=���=J��=[U�=���� �9�����@��b!��}�fY=�E�=���=��<��>�T����D��hI���gf=X��<x�&�� y� =�����
�<��\�|�C��=/-�=h}�<�f���Q�<��Ž0��;�g�������;_����%=x�x��ɴ=��;��ٽpD��4�; ��:��̽���:X�M=N�ͽj�V�`3�kE\=�$� �[�R�=P+=Ⱦ=DQ�=ɹ= �����r�s��=���M�;=G����J��|���&���<!���=���<h�I�����d�>�d��׻��^�%�����)I=�qj�< =5߀�����Q�;=�V� �8�mQ�=�L�=��$��P��P���H�<#=$��� �-� �<����i��<�ּ`�R�Gڔ�@�����=�|�=�z{< ���j��+y<�N}=]ߥ=��;�|�H�><�O�<Ȥ���&2=�gL���|���w�=`�����(���?���3��3������ ;�P���1�M����N]=�=�����J	<��<z����@�h�m< }<��X����=��8<,>�<g�����=�g�<�:����;�c�6�h=�K=�Z< �<Pn�;ie=�$=�E���=�m��i�<�s/�`~�<X旼X���"<<8�<�Ka=��N<���@�;
�=���< �=�@�<4���C�==\-�<�z�<��x<�/}:P�;iO6=���<�x�=P�7<�~=��m���L=�V	���3��ӻ0Ԃ;@H�;`���/" =(�'��.�@A��dnF=�̫;�|7;f[P=��l;�{��<�=�9p=��'=@��<��"= ��8�=�[3=�te�;��=�����E�D�x۽�i�0�� ��<��<{�y=��U����Ľ }�<�=�������<�9&<P��;�>����<l�輨�{��$��A=tv����=v����<1�ϼ���XY=*©=�B)�7��=5F)=<���8�����_<�؎=�V�=B�'=8�ۼ�K�� �m�
ڼ��C;�[6�$n���¼";=/��=�vV�LM�=^ڜ�.꼔��>���7=���<��m�(�T��������ବ�b ;����8j=⧛=W�<hV���?G<�ۗ���P���@�R;�ļ�oW����<��D�z�[=|�<��|�X�}� �2��]���j�l���@�+<2���������xN�<h$��h�1<p黻@򁼈�C<g#=ȸ7<>�<L�����=��;���= C���[�謧�`�����ݻ��˽�.<=�e���B���j�?��=8\��5���~}������=�H�����<oK��fF����3=����X3<}�T=�<�5,���$�	��/3�d9�<�N�0�L��Ŷ;���IO���Ҽ(�3��ta�b�6�s=-�P=� ��@C���愽��8�<i�<tj=p5�;���������'�з���i<��%�_Y�W���3= ��9dH���3C��ڥ:H"�&��L�ԼQ)<�b�<��6.�˕"=p�l�@����6��L=�$㞼X�N��� <PN�;�[�r~�=Ծ�<���<�۱��/<8�f<6�t�h��<�	����<�ǆ<�r�@`�`r�;w#= �~; ���`Y[;�ץ���=.�� 5ں�Z�P0�� ^��0�Ȼ%zQ= ��9l9@�*̻Z9Q=��U<�0׻����n���	�<�$<<���<�I�: �չ47���6W<:B<�j�=0w-<@�6;@g�;	=��������绀�0�d޼��/����<���@�O��k;n�=H�<��ܼhS=0�<�0RV=�t6=�<h��σ;������J=��j<�������<�a<q}��諾$SR� b��Z��=*<��r��=|i=��	�f=5�@�S��V��O���z��۫<��t=	��= �:�zt��擼���<�r���Y= 5}���*=Pzu<�H�<�3�D5�Vp�=p�a���,�#a=nŒ=��A=V=�2�D��=�3���= ?���;g�;=������ܼ4��<����Z��l�=�o�= Ka�6�X��<��<p�N<�k�<�M�<��>��QM)=�&=�哼�-�=d,�� �ƺ��X<<L�<�P�<�pG�4�����μ ��:i�=�\�=��������=4'���$��#/=�<�F<4�ἈH���νp�p��c �@ ���K��r�<�XW=�f8������X��T8�h��<؃{�(���ʽ觞<�,B�XL$<pc�=��;4F����=$�<��"<`�=<u�<(JW��g�=��{=��;p�Լ(�M<Ф�;��=Ӏ4=̍C=�99��wX=�7A���<,\���N=�e$���ۻ O�;�Xz�^3��s"=@Vλh�Z<�����f?�f+�07� d�;2�5�(�ɼ�w���<�#ȼ���:������<и<���<�Mż� ��psP��'=���\�ؼj��0&���曼^D"=`��`*ü��C�0K�;�C�:��ټhI<��h���V���,r< ����P��J�W� %���.�в��0�;,#<.��x��=G�Β=r����<`�]<QB�\ꤼ@��PR�<@k; �M<���<d�뼈{�<`v�<������@nM�t��<�K���z�@�@*�:��<󃻤�<h��<�����< B��L��� f�������9G� �;���� I�;���:<9�왃<�p���y:�S�<Bu��<�ۼd۳� �.����:�jO:x<����b;p�0l���.=�f����2=`Լ��<�r`�J
����ż�����f�;�䡺�ٺ���<x\<�����S=V2=��g��.&=��;�wͼ��w��|�h�_�\ �<|�C�lQ�=�l=�2��������D< �+����� �<wĘ=��3=�X�8\���7s��9=�Q���l�=��Q���I=XaC<�a�<��!�`�;���=[�<0���HE�<$�<O=��=�Uq�e��= g��S�=8����ջ�!<f�����2�<��弘<�$�=��=�[ʻ��D���<|��<��;�<�"Ї=Vn>rS�����`�M<���3v�=P�Ѽ Q�:�g3<bT#=��=V������k��Is=`�<˪=�����Nc�ȉ=���� ��9F\�̗��؅p<l�<��^:�q����D%ּ��޼(;��餽K�?=(�<<�-» M���8����}=���<�������{5=^)D��@<�+?=�!=�ߔ��\�=�|�< �]��!l=��5=Ly^�A�O=��=0�8�|��<���<�H���-a=��-=�0�=,+��`�}; �&; W�;|g�N؀=���<|¼ ��P���U�1~v=��O��N]< ]a;־A�N�t�$3�� ��;���� �@��:(|�=��|�����`���A< HT<]=v��r 5�p}e��%=`u�� c4�^.�x�-�X�ۼ1�Y=�����*��E��XU#<؊��RdS�Ğ�`��;H�3�T���nhR��׼<�(λ����U��|���&�H<*� Ψ���;b�D���=8���DՏ<y����]I=X-=����p &��	l��̵<���< ��;h��<���AI*=y�=t�X�@�!; D�h�<����H~��(���±�'	=@�g�R�=�6?<L���|ƻ<`��;X�W���;xs�~w�xʷ<�΁�`D4;�)<,�ü@��;@��: �]:0�;=LU� "л�<��x�8<,㹼 q,:��&�1F<`��Ȑ���qE=��(��="����=�..�����><�<����̼�	�<�$�;��D;
=$)�<�^��Sh=��4= �ߺ�{'=�<���3�(�]J��_��l�6�H-l�=�L=�:���*�m�����< R�<�ݽr+&=��=���<���U������z�<�ʟ��S�=a��~�s=����߃:J�4��4��"�L=r}=8F�xo8=�Bٺn�=;�2=�?����=��<>�=<�
��6�������\�v$N��"<d�I�C�=g��=���=��λ W�<����H��<@م:��;��b�=���=�i?���t�p܋;D������=ߢ��Hݻ�V�<o�=�ej=��ڼ��;�o��K�= /;x� <�к��S�"�=����.�=v^�E`��PS5<�"d< ?C;�mJ�\
�L������z���ǽ�̈=M˼|����;=�9}<���s1�=�m=��� ȗ��h�=�]?�\��<p��<�2=8=¼((�=̷�<rWE����=,�=�M��������=4u8�p<1=@.�"�>��5c=�ξ<���=�����*��,�<�Ĕ�Ƚ�4��=p@T=8�ͼ�oG���D�����c�=t>���}<��b<���-����)ż�$ <%����:&�,I�<\�=p�������a8���/�<P��<^��=�&%�j�}�P����=�f;@O.<�?���Ѻ������=��إ)��O��@�D;8�����n�VG��ş:����ƚ���o���=��� wc�f� � X�8:z@�<⑼�����&�.�S�A> �k�@S�;=?���u=)�=��o���;�PT�C�0=<�!= �ڻ��<l�켍s|=N�c=�v��|�<��鼈-X<�����	�����x-W���=�k�; -e=�k<���<'=�/ =�HK;ؗ<�A
;jso���=�k-�`R&<�n�<��h<h<XQ<���=���h�F<���0��<����8�^�0(\�p<�<�(;�
���H=�4��˔<DM���F=�$��R��͜<�˹����2�=p��<蓈<�%0=��=t,ż�}=�G:=��M�W�Z=@I�6����D�f�J˫�������< N;\�=,z��%U���Ȁ�"�	=Ц�<���
=^�=�<�(ܽP�X���)����:(yż��=(@���ч=����Ǽ*W��I� 3S<(�=�Y�Uf`=��j� ��;���<Pis<㭄=�J\=��=H�d�Xu
�o�� ^��;0��Ő�\4�����<!��=��=���:f�=@=��䍃�`?��x���=��,=0�� ����4���Gu���=�,�������<s��=r�d=�:���D��޽�&�=t?��Hꄼ������w�1=�5����<=r�:�g�Խ��v���;�輼�V��U��P�S�>%��@T������uT==;伌d��_�=�F�<�"_<��_=��R=���v兽��=�ؘ����<p��;<��<��ռ��j;�=����@�=t.=�]�W����=�N�P)�<F�S���=L
�Ds�=��o�f�WmN=Їμ������=�|=앮��8I���o�-V��,�w=4�ټ�7��2�<�����ȽPj �H,<y=��8�2���U=�.[=�������;e��@�
=�$=/6�=���z�w���޺���<�!<�g�<:X����<���U��=Xd�x���T���B�X�� a��>�@��;ܲ��^� ��vE��&3=��8�Мﻌ���P,�;^�[��}�ݼH���Mg��s>�!�<�ሻ�Yҽ0]`=(��<�>6����;�����v=S�L=�v� �;�p��nę=�x�=�A]��i =dxϼ kG;'k��x
��Xe���햼�^�<���<�{�=P�;��1�5f0=�>=�2�<P�Q<`iT;�ŀ��= _���:�<TK�<����3<xv;<�Q�<K��=���l��<���)=���8v�� �@�<��< ��;d�V����<�����ȫ;x/Z<��b= 
ݻt2���N<@w$<j[�E�=
i=tL�< ^$= ;=�U��X�={)=�/�g�=��̼��ؒ{�r���Dh�����%=����w�<Ŀ鼕���5)Ͻ���<@5<Hh�� = ��<�<�:3���0i�lQ��>��T��e�=h��0�=����%:���������`x�;���=�����M= Lٹ8�y�@��:BO	=�JT=�vd=���=v�����R�50���H���޼���͹��g��~��=)��=�p�:���=��9�hZ���o�,����Y�=�q5�8}\�H���Dӊ��w��R1=8���&k�P#�<t�=�{4=���P��u���Z=`j���ƚ�0᧼I����=��(��o:=�=ּ�d��0��@�A��`��[��E׻���p�� /��t�O< �=��_:1�t= K	;0�{<��<J]=H7<X������=�����>=�Π����H��X�9���<k�۽�l==��\�=뛽���=�6�#��Z�>�����p�<�y���c=xؽ"���#ؔ=4�E�PҜ���=�>
=��ȼl���:R��ۋ��8#= '����ڼ��<�������r߼ �%<�?������[=4=����%�;J�a��b�<�=&��=h�����H@\���<�S�<HN�<��"�p��;ȁ���4=�zn��?ؼ�t�P������r�(��&����;��;��$����[�=�c �@���μ@@��>J9����L򋼈嗼��;�v�=�f=��:Ay���
/= i�<��;�@��;�0\� M"=��=nf&��a��R5�GkV=��=d�?p<�Ԣ���������0~޼L����i*� ��9��+i=������� �=@J�<�׍<��)<����q���o<���w�< V-9��:(�h_m<hE<!�=�o9<�Ė<�_�|b=���˼�6�`v�<��S���I� �; J�@"�; �<C=��;��8��_.<(�b<�ϼ�A�=��6=��<ч<�_�<���"Tv=�ˮ<pM���}=�ˊ�;��<ӽ ���P���x�2=��^< �=j����ݽ�<��L,�<@�T�d��<�Ã< %�;��H� �<��q��=T��΋�/,=�� ���=�p�����k��F����<�=�I���==DD�<�����}���Ŝ<[�U=���=,Ԓ=�۽ �c�H�	�"F-��$ �da��cֽ�\l�/+�=0*�=������=�2��T^���� ��*c<=�¼��B���.�����!�@Ao;�L����=ܼ�=Y25=��*��-���཰�<`"��������A%��l*�<\v��j�,=@#��`̽ �A�����\x�~&���������2�H���佈�D� ��;8��<`��;����c�:��{; �|<Ж=����w��=}���"eG=�f���t����t�>���<�M�V�'= CB����;���8�=�C�1H��ύ����.C��W� ��<L���:��l�=�M�������
=��(�R���P��d@�j|�4ӵ<�f����ϼ�҅;�F���q�l�� ��:���n�x\=�<>Qa��v��J�H� 񆻨"d<,VN=P������h��h���P��<@T<�f� ��9���p��< �;���8샼��a;����4	�B�����;�#����D��B��Z=�^�`�;���x�n�$B���9;�`���TƗ�T�Aa=��= =/�9����=�s�<��c��g$<�N�`�p< ��;Np�p/��&=��!= ��9�ڼ�@�5��z����ݻb�h�:G�����7�H�V�P���H7\=�5�����dc�<�-�<з�;�!;�F$�s2�����;0�ϼ 6y<�����;�f3�`>�� �ͷ/��=�y�<p5��_<p��<R�H���������F<z$�` � '>� {;�1�l��<��.=8L�<l�0�	<�t�<���GP]=�|= �-��]'�`�(�����2�=�*:�� ����h�ռ����,c�dw{��J=���=�z��%��=�T�=0ƻ�N���r�<��;�F�S���v�:=���=È�=�<e<�#�t$���?��0"&��e=�D�;[�"=�"�JJ-=���:��=4JJ��U;=�nZ<�>=�Y�=��=�ȶ��y!=V���fz<0y�;�2���=v_���K��G=�Ѽ�:ƽW!�=�ҝ=�3�;Խ�V��tHQ=�*j� 	�:���<Vk">�	���{<���<�㕽}5�=�Rl�`��;x�<�P<���=�B�����[���i=@l<gV�=h6��F�m����;ĊJ�d�ռ�O�*��=O�<#5S=��;�"������B�@�<,��/����u��=<��h;�t@�7/��1-�� *�<�#��@1E�ڿ���=QЁ�8�<�
�=2k� *w9�*�=4��<Զ=����<8���r��d��=�.A=�K����7<�I��Jc=��<rO=��i=F����==��u���$; ��Ȁ�<�r� !�9�{�<��B�>�M�r<X=�񆺠";�N�����ZO��[-���<0����P���%,<&�@���<6�.�>{=P1���=����,^�<U��t= ߾<���0U�����<�|��} =��k��q;P?�; �<�^R<(FW����Q\=Pe�� �0��A<x�=�D< <�9N�e� �ɻzm�D����>���+�������= ��< �{<*>1��A�<@�v<��=����;+.<؜=��<�}ȼp��B~#���F=*�=�%	���܏�<�%�<!Ӕ��������;$�<�=����y�=�������;A�i=@�_< Ղ9_62=T毼b�s��ގ;ĥ���r�< ��:L�ܼ�$�;����3g<��	=Ps�t����t��fC<|M�<������ռ��=�_�p�ƼX_<���(�.= ��<��< W�:^�Z��9����<����̧�<��<.<<�!$=4��<x�Y�>��=�h�<P8���Ƽt//��X���0P�`��n=b����%=�#�=0�ӻ�54���;�n�< u�9?��N�@=��=k�C=�_��54�`z��0R<�ʼX��=��'��J<�����=.���mT=R�=8ӛ�F�= *�;�ߑ�J/�=���<�yc�`M5<�(�����< �����@����q���!P�,T=�@ż0�߼�>M�q=5��Ƒb�pj<@=����ȵ��@P=iz�=:����t�����:����v>6������ �6�@̯�E��=���������#��=�k<b�L=8l,��6!���;Xt|����#o���r<h�v<s==H�!<Y�N��p&��0ʼت#<n(]���<��̼��L;���<ZM9�b�D�r_:=lt�<�p���l��x�=�都XL-<�(2=��=ؿt<:�=���<���<h"<L��<����E+=| =�-W���=`�;�/<(�V<̃�<-o{=��]O<�维�<�9м@˦:(Em�X>��P<dlʼ��a�{_= E�;P��;@�ͺa��8�������5� $\�T��@�*�K�=�(��c�<���Ѯ<��M�P��<�y��� T<H���$O�<�A=|��K��ڄ<�a���Z�<P;Ȼ�W��h�<���<Ŗ<H����J���=$����x���܅<4N�<��6< x}�b)N��\廾�%�<K��B��U��� � ��=R=��(;��)��\ =�x�<��6���<�<hN�<��<����8x��,�hY=�<����p`��$@�<Ҏ<�m��Ғ4���:<�ˤ<��=�߻p_= ����-�<3K8=�Iz<��u��cp=�!�r�R��u�;>��p�<0>�0����:�h���z><DT=��� �9�{�;��)<��< ��9l�	�� �<x�"�������;`���m�[=���< f
=��5<�F�(&���]8=�&��l�=p)<0=�;4I�<���<X��[Q�=��i<�h�;0�Y�`�I�d.弼����"%��A����<�Z;�F���v=g�������Ǽt �<`�!ĉ�H)D=�=��<ۘ��P�^�0��;�^<0O��!ͭ=LE������D�; ������Ȟb=�.{<P�:< $0;���<�G�hHh=��;pŽ��»�?��=p��3S���6vR���e�?	=��� �q��p�=�=����0℻���<|N�<�#���0�V�a=[+Q=�����F��$S��1<���=�;]������=��b/�PY=��Hб�1��{�= ��;x��<X��C�� <p��;�a�EG��t��`��;�h�<�*��d�<�
��C_��f,�a�B=/d����<����KĻp�=�4��xe���� =��=}�����4��<��W����;�Ȅ<�"�=L��<R<=\�<�|��8<�$=H��|#����><�zS����= QA���"���� X�9[D=pż�@��j<Py�<t��@�<��m��h�`�<x����m�_c?=PN�;P��;���� ���� �����: 0q��7�3<|��<�90�'
=�Լ�<@»\)�<@_d�`��;PVN�|��<��#=ب0��d����<@�x�<�x�@���0�'<4��<���<�}��=�S=c��x��<X��<H5#<�����@z���*������мЌ��D,ռ�b�=Yh9=��b�/�4!=(x<�f����<�'�<�Y=|x�<�O���y��|��*+a=���< p��PG<t�<�� <����RP"��rb<�K<�ư<@� ;�=u=@9���V[<g�7=�m�<`��;Tjo=(��(��'K;��`�<��9� ���i��h�L�$.�<�No=`e@; �< �~:'J<P=<ЙĻ@$�&�	=���Ȃo�P��P�»��c=5V=� =0݄<���ͼ�lU=H͐�W4Z=��<��;4��<��< �h*�=0�;@W?;���:\rG�����@�����;@����m����9:8��`y?�d���,�ܼ_6���ި<���旽_|=0;A<�R7������3e��|�; ĝ;4ɼ?�~=,ր�*8$�hsG<��B��������<؃#�l+�<���P`�<򼐸�<8ԥ��a<�Z���L����T=Po��������ʚp��-�@��:@#���$x���=��=�H��UK=x�<`-� ����J���b(=N�,��� <�x�0�ٻ��x=�̚=\>����u�p������(��<�S����[I��ȹ= DJ9 &ɺX
������Y<�M<��3����,���-���H�:������=�&_�4��x�^�?j=����X W�Ğ�����c:�=(���<�s�<L;�< ��ı8���{<�,���t;���2K=�F�<�i'�0��<R��L�:�t0=@�!��X{��o�l�1��6�=4P���{��F�t�\��V�<��x�����Ie=(�1<�^M���<8�D��17���#<��P�1���=�L���%�����k��zBh�JG��N�: Q�;@]$�,��<�)��T��jT=t젼B�==��<���<�7�� p�8 ~!�P�o<�=�~�;�Q��%6=�,L�`�<^"<���:��; Ay�,��<87��R�,����<P�%�0��J�=�\�<�ي;\򍼞R
� �����;��A������
�B�����=�8W=�мB�v�Ф�<�W��`RŻ���<t��<}[F=,1�<�霽�v� �i�ٟq=��<h�4����<4�< �/��������u<@�� t�9�p�;a%�=��Z;���9�X=@�<�.E<�%R=Vm��>�pn��27��ٻ<$���w���K~�(r���l�<�=�j�<h��<0'���a<�<H���d�8�= �%�(��`Y��D����o.=5Ė=ln=�-^<���� ���O]=�O�f�=�+=�G;xCH<���<`'_�W:�=���;�^��ȋS<d�K�Ȅ�<@=�; )*�4��<�gO���=*#��w�t0����o�����7�<����I���y�<@{b���Y������G� �� �_�l���/=42��ʼ@���wi��`�M�D뼔^k�z�=�n���<�D��'r��Hy�<Dwʼ�`��r��=.������Pr��:�t�������l���n=�Lc�<��{=��n� �=��Z;�틽������P�=SC̽dn�<�G}�����V�=(�/=@㟼&����՛����$��<�o��R7������=�b�(�a��Q��M$��X��<���0򼏞���޽��4�lf޼U�ý(�=�� <h��� Aw�V3=�¥�Ω����;��;9¨=��a�T+�<`�N�� �<��J�d/��	�<��½@/};�����jE< ��;�$�����<����p� �P�U= ����3Խ��0��'&���<JT8��ǒ��D��`׽���;K?ངo2�dR�=��Լ��5�8X< ݼ<6����_�ZSr���< �`�DO��@�z�@�!;j�� �8�@��;�<����O�<WH�ψ���H=@�����<�<P7^<��;��`<������
"=�X�<(f[�Db�<�q� �(�hJ<D�����|�@%�;�]�<�qԼ+��0
<�&<lU�@�<\��< -�`I��nڼ ��>��PJ+�0��޷9��/̼�=�r6=�����a�0]�<��E;p������; �}<0��<h�<a-���}�@����=���;���;�����;D�˼O؄��.L��JC<�.�J������d9= �߻P�M�$�=8v?����;'Y=f J��9T��i�2��o<\ʼ ��;(C���P�8�<��=(ھ<��;pK�;���<�৺�� �G����<$ �� �:�$�� %g����<� �=T�=�W�<Hc���4#=����}n=��<&�0˻���; -h;>{�= h���[��vI<p'd��B�<�k<�C�a�E=hԢ���=X.��~˻����>��T9��
�<jq��i��,��<��1<8�0��U� $.�����,c����ɛ<��7:��<�.$��˽ 9�ɍν�|��A=R0��@��;0��;8���,$�<��=�l~�<`N�=���� !y�Tt���+�:�`�R޽�fz� |i;Y��=�i�έ=���W��lf��f(����<��ӽ0k �`h�h�<�ը<���<ܞ�f8ԽH��<@[o�0��<�E'���)�ֽ�W�=��}���u������hi<�F\� .92�[����H�)�.�:�y�(-<�u8;zm��Aw��k�%N���UF��9J<�=6G;=�����#<���p�<�y1<�+]��w=S��8"-<�ʲ��� �������=�<#�ɽ�Vj�h��<���<$V�X�x<,fi���9�����٣���J[�u���ɻl���9g�n��=�}����ü@��;�Bܼ��B� �¼Ȕ����r����;�j"���� �x��C,<�X<��B�� ��; ͙��+��=��'�g���-<P�'��ȟ;H�M<Tɡ< �; �?��0���6�N0=�<�PǼ���<Pjռ��s�T~�<$��(n�� 9 ҹD���$���8����;2D?�P]<���<`�ջ0��;V��(�O��y� <Z�4o�>b>�,�����=��=Zj����7�=��<��@�; K���.+<��=�)/ϽL��`?����*=_���;䍌� xt��$��OW���H��8�X��*"�0X� ��K=H1��޹����<P�
� �9H��<>c����\�м��y�P�;�y���,<<0�2�����Hj�K?�=�7�< #���H	<h��<�s׼\������+�<,��8=�>��W�:��ܺ�h�=��+=P��<��Ǽ�8ռ@P=�ж�|nt==�<<ܦ�h�Ӽ�<r��P�<2�=����g��@����J����* �`�,��o�<F&�=� ����=59�=����IX<��H= � X�:��C��J�<���=R�=�:�<H7g�Px5��;L'�����<�J|;pӁ;�;��D\=�*���td=��=^���`5�= >z� ������=��=Tm���O��ս̞�P��<�帻��<@C�:v�y���=���<rҜ����=�b0<$�	�X���<�8�=ĭ���
�:h�����>����f�;���<�±�PH >�!ɽ�m��hu(��/��ֲ=f�P� �9:2=�v=���<ɇ�=$#���L�<�����4�dz>��O�y0�= a�;�ѐ=x�'=p	�<}y� C-�Ү< z<p[���|�𽈼����ߑ�ͽ���{�� fJ; �|�@�׻�ή����<f+c�ԡ���A= �@�(�<��>�=⮄=����c߼ē�w�)=��<®����=8�`�� �= ����p=hQ< ��;��n=H��J1=��o< �"��l���E3<�<�<�
/�E����]=p�󻀟�: w��1J�#y��@���Ђ���F���6E� �к��t�>*g�m�w=�hh��CK=p����y=�pټ��<@���+�<���<����Tb��v�=�v����=$�<$M�< <�6<�#=�/m;r�>�j?=����`���Q=��=(��<����ʬE� �a��8�P����b�������6����=$��=@F˼��D� ԅ<���ʴ!� ��<��s<<�$=D��<^�R��&�����=�4*=t�����<�=0��<���.6���<�ݣ<�9#= �;���=&]� �����t=T�6=�
�;���=>"�/刽�wA;
��t,= h޹R׼`�s<0���P=��O=���`c�;@��:d$�<�F�<����ɼ*s=��; �� ȧ��� �=��6=�7==���t�hB2����=��Pj=�u+=Wu<` =�>=���;:��=`&Z< bz�t�� %d� �:�?	�У�;�$2����=p��� \�8	��=p/��P�=�-�<`��;`p��X�M�\��<�Μ=�=�	.����X�;��<x��ҙG=@�.�Τ0����;�n=<���h*>0�<^ٍ�;�X=��U��y�s��=`�;�D3/�w�\�Խ\Ӕ����;���@���h��j�X�^ү=��<x�&�tC�=�9��
B���ӓ=�(�= ?x�pN�; ���*�=|]z�����a����[�>Ͻ�;!�2�Ih����u=b�2� {�;��%=�/i=z=��<=4#ڼ(`���K��|�;N����{��F�2=@A�6��=�=쌠=����<�>�,���ƥ�=V2$��^Ǽ���PTV���T=:mO�X����<�L�<�߼P{� s��J�?�֙M�X7@<��e=��`=��=�.=+In=ZJ5���<����/=�2���2�E�=�Z�<�F{=�)A�`=�F�;�<�=�a�H�=P���)��|μ�H<��)=č��}��p�C=�; ����/��R����ʼHz<�����E<�'0� �:�#�<V�H�}ք=N�3���.=@�B�|��<��s��@J<X�r�즜<�)�< r��I4:�v�<��8\�<��<D�<(��<�ݶ<�BV=@�㺤���W.V=(�i����;	�v=X�<�Vk<�-���k�E <Z�3��W� 1����e!�Յ�=���=�������;����d�߼��:=�3�<�%=��<}6��G#�X�V�-�=ϵ<�w�;�[�<�lU=,�<ׇ������!=���< �<�};<f�t=��L� �R�R�=)vk=�<�i�=�:�^B#� �z�j���9=亜� ��䩃<�_����=���=�:-<x�=�X-<`��<��<8���Ǽ&a=P�;`a��0o�������d=I�m=�7=`��:�E���Hl�= ,:�ph=\��<П<�e<�u=�A�<GV�= J�pM�<�Sb�X�ȍ <���xHi=�:���&�=<)\������W<�(b��(=�3��`�<��-��Ψ<���<$:�< FS�:���½�,=���<`��9&=r�����P�3= p�8����m&>�?4�<lL�,��<@��:���̋�=$|1�Ϲ;)Ҍ��4ʽ0���8@t��T�:dDȼ"�*�݇�=�J�<��M<���=�瞽���;�餼���=��<�΀��a4�@WڻRB�8�&<�ݎ�����=#��=�½��z��.F�%ٽ�a<����a&�4 �<�Y=���<P�{<`E��X̥��o��G`=dD����ѽ@]q;|��^+=n໼)>��ۼtz��|�Z�i�>�4g�Z��F��¼���=��)��ڼ�y�<���<��7�T��$�2�zI���G��&���[�=v �=X���+3=!"=!쐽Cir=()1�0֏;>Խ�����=�J=.�=qh�� ��:�.$���;�h< 4Z�9��=�ZT��դ;�և� =�<��B=��+�����er=��D�`�ǻ8����Ƽ��
���<�f��x,�<��9��@�< *�&vw��t�=&�4�=�w�;p�~< 㥻P� ��d:�<_�<��h<  L�n�= xM:9*=��J=O==BQ=�p����=�� �^�	���h=�7����<�g�=���<�,�;L�� n�H�<�b�(�b��z���:�vJM�rZ�=C�=�[Y�^����,)S� ��:jxD=b�?=��j=4:�<
x�L]����;�C�= 5=�ø<0#=�Sz=�ϱ<����h-���C=�X(;(U<D�=�l�= �P;L���� =̱�=�a<�:�=����V���t��ݼ�6C= uͼ��U���?=�x`��e=O��=8F�<�_D=@ע�l[�<��=��� 4Լo4T=(�]<�懼�����O�Ȩ~=$��=VG!=��ѻb�#�d)I�@h�=@̉<X�=�A=`|�;�M�;|Ȝ=m�=�9�= ˓�H��<r(��/�T�<�9.��־=���<X�\=`м<��N/(�d���pG�<Mt��  +���o��E�<���<����䚼�?P�P6Ža�$=@�<܇$��Հ<|dt�F?��	^=�0� ��r;>�ֽ4M)����'�;)%���I�<`ֈ��� =�I���\��@ 4<�.c�.I���� <jV�|z�d��<�. ��/����<򨚽 g�;(A�<��=�[���Z��,����b����`ym=Hyۼ�t��r�>ks=�_{�����6���K���ݼx�ؼ�38� �r:�|]=@�5<����+��S��9��;�=�� �Pѽr%�����y:��w�I�>H�<�q��Dw�;%>e�~
P��V�ʼ1f�=w�`<��;��;<�G� ���틽n�8�u �l$	��be=���= K޽*-=(Z�<sp��aW�=�1l���7��w�丹=(��<���;�s潸I8�辏�������;��<�6�=psp�\��<��'����<�B=�ڷ�����n�<|�ļP����1��4��o����<xln��5�<�Ur����<\T�������>,��L�=�:�<h�"<@�?;�m���:/;X�K<��B<��=<ټ�1n=����hj=b��=�=���<��*�`��=p&�X��MC=TD��Φ=���=�<�<�0Ի U� `<���<�����ۼ m⼆E{�����
��=���=mץ�.?o���ͼĒ��huf<�~
=߲A=[�=@=�n���Ə��#k<���=X�8=�E�<O�e=�Y=���;�N꽨)l<�=�nȼ�� �&s;=	�=��<(�����U=�՝=`؀<�գ=���W�������μ��1=X���T&�)�=��{2�=�<�=%=�4B=��ּ�r<�=�����g�JZW=8�U<�M��N'������AQ=k�=p =����ji-�4㕽*Z�=Ѐ�<���=�-i=@D=�`�"�h��=�t =�#�=�ˤ;P��;����=��/I=0��;>ͭ=��m=��;D<+�?ր��@�� H��Sbڽ B��n�s����$�<fh�h���lf�6v��Pe�<�֫:ЗF��������b=�t��<�D��:�<�X�=3����ݼп$������R��$��*c�N�	=a祽�v��p��<d^��PA��`,/;�L�dX̼ ����
E� (ؼ�N��'���ؒ;~�b=��=�:t�̒��lh��Vf��<�rZ�=(|6����/�>\F�<�e�6��L�������lN޼`�����}�&��Sy�=l��B��p�ϼ��ͽ &d:�f=����E����½l9Y�n3�_ǽg�=ܾ=����l�,^�=��L�`���;�8!�K{�=����L)�<�,����
<p�%��� ��pL�C{���ڼ���0[<�yh=h��F=����yY���1�=�Jy<A�ٽ\
�@��7�g=��Z�K �S,޽�۱�U��mך������=��=P�E����<䣎��aM;h��<PͽM���й�;d���D�'���*��T���u��~;@��:��;`�j����<�ּC�ڽ6��=��';�=�l%= 0P8�sr<L4��`��@�)�H�<2�8=�U��N�@=�\�d�<SEo=���;x�}<�B�ص=Y��Z�1�h��<P��;�A<�h]=~�<�W|�@������<��������Ǽq���Sʈ�k9�=�Y=5J����Y� ����Tz��X�< �<�Q=-w=�z�<C���>o���<�rj==��?=Xq�<@j�<�ku�ս����ʒ0=�q�T6����<U��=��P<Z7����(=���<8Y<x��=J"�6�V��R�����<>�Р.<��T=�Ỽ�b-=-.�=F�<Է�<��꼀��<< �<N������lX5=@DJ;Ц��p�V��q�e=���=��=(O��0������4�Z= H�<v �=)�=k��P��1:~=��=3�= ��: _~�ص��&e��v=�h�<C0=2��=�6��`D=�,ֽ�3(�~g��V�r���� <hH��3F����< 嶻����t��[H��H<�8��TzE���L�4�<4��� ��SGƽ`9�<�n���ٽ@d���~��d���6��eڼ����<����м7>=n����̽����F���������[����!��0�����<Й�;�_=@}<R��`U
��(�� ���,� �&=�]���<;jS�=��<@�¼�Sѽ��$;[ý��g��k���[N������=L���pV�p��������;�9�;0D�te��>���"s���]��mƽ�Ղ=��=B���,�;�J�=��A�Zi�P�;�Ӄ<.��=`L����<�鍼쫐<�`y��O�� }�:w�˽�ꃼа������`�<��Ľ4�<�V��>>���a=\�.=?"� ��01C��<�Y�
�f�ӛ�� ��,&���A����ݼ�A�=$t���¼x��<`��̨��h'u��$��I��� ��9*m�p8p�$6x�XE(��ֶ�`p����A<��� ؔ�[z8=ԡݼ�
Ϧ= �d�'hu=��X=��?<���Nb�.D��S'���<��[=ZGI��[=�j�П�<�nn=<���P���`>)�$��<N*��KU��< ,���]��sE =�\!=��R� H��@��Ȭ<{�dD�t�*�|���^�����=\�?=[Cǽ������<D�A�xP<����
m<8\=PT���#�f�\���M;���=�[=l!$=���<�A�P�>�䙼`�H<P�1��C��@`:�~�=��*��ė���%=�}�<�X<:L=n�%��췽8S_�F	x�d��<ޤ��iI<<
�<�Y%��<��=3�<�{�;X��R�=@��ǽ����cR=�F�LB��h�Q���e��H����=�G= �O��
%�0ˀ�+rA=@�n;���=s!o=>����-d&=$5�<@��= �; ��.@��	j��4d<<YJ�H�J�pgW��M>�ص���q=�=�=<ݼ�S=#P�=hwS���$� ��ڡ�?y=�7=�,=�-����V����<p������C���T�в�;��l=�A��,��=]98=�ս�Aw=2��Cq����=��<�=���Qٽ��`4v<� ;Ȑ<�Qu<*Ml��8�=�i�=��꼰b�<�b� ���"p ���=t��=��ܼP�Z<�nN��=�=������];PC�<n���o�=S�?��R/��T��G�=�qj�he=�\�=��;�K=LU]=P<����im�(o��k���'-���=��`m=V=�I�=���Ą)�\�<���=D<W���Z��S˼ KR�ƫ}�5ó������/�<P�Z���0��+�;�3�#�˽x[<t=9�0=9P�=�n�<)��=6Yd�8�y�:���Dr�<駼(�C�(�<��<���=r3��u�=�4�(Ol=�w=(>>�؂�= �:�V�����:� )<P� =�(�����R�1=pތ�@`<@�: �A9-4�`:5���� <������O��;\����r=�:��|��<�"���<0�ؼ�廈�s� ��9�_�;����E��`U�<@9-�P�< ��<4C�<P<`8V;]�&=�Ç<|�ż��#=��`|;
Y)= *�;��<�� :�t�@'o; 4�������R�|�������֯=]�d=^�B�6�/���; څ�N����< t��Զ�<h�#<����\���c�Yd�=,�<p�W���=��=pd�<�G��أټ��z;@�R<@E7=p�b<��S=xP��@>^����<�l�=��Q�d��=J���8����<:���|=�;��p�i<�P1�`g�<�O=�%3�\��<��<@%E<�k�;@/
���W�ȿ}=d��< �ٺ@�f;p�ܼ ��; M<�;V=葉���N��1����d=��	�})=9�=��<Pԇ<��=�N�<d1�=��I<\H�<��ѽ��[����< �B�]�=y�4� >v���������F=̹̼��^=��b=0�b�*�?��zL=TM���<0R<�b+<|,ʽH�|<�5#=X�3�8S���~�u��ɪT=+'=�����E/>hvW��^۽�\=�]���#��=�������c��.��B�p� :;|��� z�;(�f<��"�5�=���=��<�G =����l<�=n>�=8�a����<d�r�RK
= ��º��<��,; �=)�8R[�ǈ��[��d�<�r����<�=tԼ��o=D�<�r�\��z�b�H��<�9�Ew���J�=@Aм���=Շ;=�">i���p�@�P�ݼ��>�Ʊ���,�r�&�ήI�u�'=2X����P�Ż0��<��Ƽܷ��tBQ�l��)����N��m�=��=Sj=��=��=98��$��<�}�`)=(����
�~\�=t�= d�=�*���ˋ=Zch��ʒ=�20=��:���>|���M������V<�X$=�8�<H���=4ؗ<��$<��p<�I���� <��<ގ�pW�<��9����Dp=�"���=V�\� �7<XG[��z5<ܹ�� v:�뀼`�n���s;p����	<���;@�8<p/<80�<�<��=���<��W=p�l< ��c3=�w��(�8<��7=��O� 
�< Qp:�a��0)�<�l�;���<�Ϡ<����<��b�d=�o=�x]��OD�P����έ����&2/=HIq<@��:��R������Ҽ &��yԊ=�
�:@�o<�U�<��F=���<�������F=�<�=��<ȧ�<`�� ����(w:x��= �W9��=�g��n�T��<�a޼��;=�27��Տ����<PE�����<��\=��j<?A!=L��<��C< �;�ݕ<@N�o�d=���<���<@t�0�h�8�<��<T�;=@$��D ̼`)��lQ�=�Au<8v�<@��;��'<P�λa�M=~eP=�f0= ���WE7=�a��\"C�X��<��F����=,-%�uz>�����!�81p�
-�?��=�<�?i�[T��N�=���p����ּ.6���2�y=|=D���TT�j%ڽ�	=�E��=0��;X�(�yz>�����ȽJu=1�����)�=�1����A=��ٽ)�	���\��������;P��;����|�=.��=�"�<���<�<8��<�q%���/>�\=�3b�x<�`���Ƚ��L=h<�@-�P�=^�=j�:Y��2Ͻ��5�����G� ����J�=01�7�=p=��\M��+ռ��0�;��=04�Pν(!�<,�
�Ģ8=@3�:��U>z��Xm]�f���B�9>��\�0�����Q͋��=. �.�1�`�;L�<��1� �
:<_�Z��ܽ����=���=�A��X�O=/S�=$���ɢ= _��P=��@�X��<䩿=��=��=���D�<���� I= �<@���J>L�Z����X»8��<kl_=`�������!�< .�;��� ��8PWU� ������<,���Dm�<�wm�r�<hq<<-˼���=�V�c&=�-<��<�0��V �p,� ����2;\4�<PK�;��<�K�<�!�<��K=i�=)=�I:>u�=(yd<�7��a= ���~�= �y=�<8a<�B����<,�!=x\�H]�<Dr�<,(����=Z�==
���4��4�����B��x�@,E=W=���<@?";�ˇ�Z'0��x�<�h�=�<�4=�P=l�r=H��<J�E���λG�%=���; ��<�X=��= =���f'��[<���=��<���=� ݺ�@�:<�~��:.d=t@���<�M=0���>=@ې=ؔ�<:c=��J<���<l޾<8���Z��|m=6@=X�<h���4�$�@u�<{�l=Q�,=�lH�Tc��􂄼U*�=D��<��u=|<�<ظ!<x��E�=�=[�J= k;��;=<Rf����O	=T�"��>�7���F�=�#��b�e���{���)��l~=������K����~>�.��w��81�Up��?�$�=�O�<�R��5���ǽ>Eg��Q�=lQȼ�J.< 2v>r'�����ṕ�L༴�Kf=~桽.�r=_G߽����e$�Ԍμ"*�����;8z ������e=�y=Ps�;�����*�@�K;z�;0�> ����������n���E�μ�=@s�����@M<>dK�</-��|�p�/F����H�I������d�Q��79=�@��x�/�.�X�)�~
:���鼓�>�^3�C!Ƚ"��h�)�`9�;PD���=>��;��J�ֶ��CQ/>�8a�Ł��H�A�G��r&>|��0活 �;�)˺��l�@�G<�C��v�3��'����G�b4(=�)>8h/�-�P=��=��%�;P�= ����*�;b�y��>D<k�=&��=�h=
!�8b���bd�p��;��<���<3<>�_�+�< 7ջ]�!=MZ=�t�k��t�<@�������|���߼�<���<0V��pE�;І���=��g�S�����=��T��c�=Ϟ=HZm<�FԻ�$� w��mŻ`p�L�=��֩F= �+<�S=�5�=�%+=k9=�9�rڍ=  �n��pR=<ڣ�h�6=��=�ߝ<`Nr�t'��y<<V"=V�+�@��;pG����!�U䆽t��=i��=cٻ�R0��Y�#����� =݁&=�Mx=�c�; �?5���
�<cȯ=��3=^�=Wq=�U=P{�<������<�J=�T��`��;V^4=�t�=x�,<Z`���g�<D��=`2F<[�=(W���ἰ��@s����`=�ۍ����;bҘ=P/��	��=d�=��
=)U=��(��R�<@.�< b��n����=�^�<�Ӽ����t���&�<x��=��2=<� �V��7>��q=x�
=/"�=�*]=��;��&�n�=h�p=��=p��;��=��e�\1��::=8�b���>�9=0�U=��?��oX�����Nu+��=�g}�(�k��+�����=p����Rf����#���������=��<�lFw�����X�C�,�M���=&�e�|�<�*)>:B��ߌ���5�D����v���D�={.=�!���rݽ��:<�|&��ɽ��������.����<P��;��%��5��̱Ͻ$�����<�޽=�E?����,͓�(O<��}W����=�5>��)����=>�W����C�,�{����tE1�/	���]q������ef�=Tr���@� �޼퇧������=4��%�������tbG��j��'G���7�=�x@=�Y:�V#��zQ�=F�:�|�����8]���> <���<��;�����Pm� t�;tR���cY�6�!��5A�0А����=��2�gvG=-
��n�����=�l<���^��Ц����==�D�f���FR�v
�����sI<�'�=��=8!�|��<8�m�ؔ�<���<X��z�z��w�;0S��*����.�@o�� zX< ��; �ٷ���=�3��m���c�=����,'z=��U=P��;�_Z; (]� �	����0�s<�d!=!���A=��*<���<Ö=��T<<ؒ<h����r=�sf����li�<��;t&�<�k=���<]� ���4:<��<&yL� �ܺh3*�>�n���]��$|=r=	�ƽ����ͻ��e���'<��u<�=46= ~ѻBqj�z�;�4�<W|=9�7=��v=�~�<��<�Ռ;u���@R�:(�C=hw���Қ�䙗<��=�c�;X�x��*�<r2=���<l�=���C�`�J�b�"���:=@�f�Qo=%.q=(�~�ę=�@�=���<�y�< ��� �<���<&g�����~�z=�A�<��R��xO��:g��ё<4��=O,.=�7ʼd�ټ pG�N�+=H��<�ÿ=��=����@U^�=�= �5=�ԡ=�@�;l��<�i=���L�H�Z=P1<��=DÆ=���;����{(���l�h>�p�\<]�ʽ 1�:pN�J=8�:<d�Ƽԟ7�qҀ���N��y=�����`�������;����R�I>���v7=S�=7�
����d戽�P���������(�ۼt� =����6��� .�<r����1Žp���V�8��Лa�8S	���伿aŽ��+�� *�c�<��5=T�~�(z-��ؽ�8	�fA?��Q�=4�e�\O/���>�x%��B̼`����p�� x��>����et�:���2�=nq�$�B� _���@�;�r=�^��:4������B���P�����V�=y�a=h�@���o�'�k=�V��]���)�L����=�P�<(�< vF�@�(<����YE�Ģ��,�������� �Hc+�|��=���>F!=Fo>��2����=L��<�v���������*=(1�����uƽ�m��|����� ��`γ=dy�<��T�'�<|�`��; g��#Ž�В��ݻ������&j��Љݻ�������X}�<�5��m]�Ad= ����G��J�= 5��7o=b�=�:@<��6�Pym���8��c�H�<M==��+�WZk=�����'�<�m�=�g�@5���ټ���<HKڼ"��84\< �)�h�$���5=H�;=�3���햼�PI;��<.2`�lEļ4�7j��"Vy���=�h=��vm���<��6��D<PE��୭<�g%=HJǼAE����0��?
<Tm�=��U=na=p�.<��컰� �=��0p&���<��-���� 0���ķ=�#�(���= ��<��<��U=�XּI9��@Ո������=@-����*=��=����i�<�P
>�/�<�t�;��ܼ��=�l&<L�ν���rs�= έ:�߳��+e�Hsl�h�d�L�>� O=�㼒m+�(�Y��=��A<�@�=�j=���� d���=x�<To�=P��;�?D�:���8w��p��<X�����o<|���>����=b�F=�
� [�<B��=���#���=Jn� ��9`��;2`J=�3���r���#=��/;��L����
^N�^�A=h!<=��8��=�,�<�H����=^*�;Y���J=p=�<����OO�,x���������*�;�\�<L��<6<����=���= <��Ǽ�$����<�l���=���=�������:��^�1O�=���K����(=r���\S=��t������۽�H7=�Ǝ��n=B�=�j��Af=�ٵ<J=$=���������#�X���f��X��=<��D��<���<�l�=����:�p>�;{�g=��;x㓼�,��Mм�#�5|�������S���=ܚ��`;�8Es��b��8 �� ����=R�=⩁=�~�D�h=�~��`TL;��L�,�=T�ɼ�i� (h��h�=�j�=*IK����=�ס��|�=K�c=2ҋ���=X�޼`b�O=�x<���<�7��X��tb=`_�5�<�@<\S�<j��𻎻p�Ҽ0x���2R<�j�<(�q<�t�;K���}�;�P<<�=�ռ:�@�޻@�o� #`�@��:�f��X��<�傻��<`	�< �l<�Z�;)C�0��<��<@Sٺ�w�<���8<�8�<p/һ.A= AW<��9�H|b<\*=��; _ܻK<�F����=��=.8I�FY�`�;@T%������<����Tt�<@Tͺ;�����S���f�=|��<H�T�p=X=g<�N�<Ri2�����������$W=D�<J�=H�J���ļ@6d;iE�=X�c���=h[Q��Ѽ
M�=�����<�Ϡ<��ϼd:�<�c���=�B�<@��:�ׁ<`��; /<�e���o;�-�;�hv=�D,=�<�O�<(m�HKݼ��ۼ��{=����Jż�5<��=�Ä��q=	5=��< k����<���<�= ��<	!=,���вy�D��<�����=�B��c02>td=�l�E�XlD<��*���7=<x�=��/�8���^�=WӍ�fA�����|��<��н�\��G=``=;^������ܽR��=�`;=�*��V$> �׽�;=J�E�#��0σ=$ ��@.�<�����b���͠�X�W�(�p<�=�sE��/�=��>��=�׏��� ��E�<�=�*�/>fN�=��k�t#�<���l<�!1��*� �<ؤz<�@=m��T�$��d���0#����6"=�+>Z_Խ]�r=�m�@��<]��������<X���g�0�=����e=�[�<�?9>�@ҽ���-���=������D�����y��i6<B�v�Q1��t���䮼<0���K<�ߤ�Vc��K���ɼ%�=xC�= ��:@�;&~�=����	=�,J�ˊ=�Խp:�<��=n��=x,�=U���8�=s�սL;�=ܦ6=fݖ���0>�;T��hż��<Ж�<���<���< 	H<�b�<���<��z<�P�<Lj�<��< ��<&��0�q<@p<�U[<a==��<�ۿ;
�t������=/<l��< ͩ���@�O�Xs����$��(,<8y<��;ء�<�)<���<�"�<��<���;2L)=��<�+�<(
=��w�0"7<��<�e�<,=�J�<t��<��<�&+=���<�J�<8�< ��:��C=584=.�I�L�����q� x�8��s�= ą��b^��%�xt�<ؠ� iN;��=@�s;�jA<��C=��<D/�<�4�@�G�0��@i<�s+=���<�u<<�䡼X^�������=(�'��%`=8J�`0<l��=�0O�E%6=`�< �;:l��< ��7�
=,p=���<.I=(p�<���<0��0%�<@<v<�g=W5=m�9=P�<`��;h2���C���m= �8;�mT��G�<�FC=���<$��<�7�<`9�<Pq|����<gzS=�`�<|`�<�ER=��x�ȨE��c�<����L�>(̽��3>p�l���O�Z
B�^�6A�=��<=N��q\���,,>�'j��^ͽ@�,�aX��齁�Q=(=缨����A��M��		>�`�<��O�]��>`��o�� ��<�XY�t�.�dr=ܴ����=������~�=�4Kļ�[T�0٣��9=��<_&�=�>��5=تq��g��ex<�'�$�C>���<���P�W<�Ͷ�y����=�x���Sc���>�N<������5��\&�x\�_䏽z�n� ��� f�=�\��3�<`�f�t�������b~�$�
>5\R�QϜ���<��(O=��@�2�h>bZN����㽦6>�y������>��AϽ8��=�����Z<x5<$?Y���<��֨R�����lb�	�=9.>����~�<�W�=V�@���=I�����=F�Z�t�=�Y^=#�'>�f�=
e�`��<�7��7�=0r�<<�S�D8�>���P|	�� <�=>}%=E�;����T)�<0��;P�*���<,��<�i1����<���Ȇ�<O�; ��<�=@V#��N@=Z���<���<(��<lX��@�ûA#��&���,�$��<��;L��<h��<��<},=Z=$�=�����=�ȶ<@^;T�K=,���h��<;�g=��<���<P٤��U�<0}.=L��<�#=�v�< u�:X�n��h=$��=�z�J�\Hͼ$��@x=�65B=@Q<��<`� @;��ۼ���<�"�=b�<0��</�p=�v+=�=�<j(�@�:��;@�];��<��=s�=�����`r�F.�= T��=`�����<�M=0�R��z=`ge;8X<tMS=�ש��|T=ډ=��%=��;=X'�<�*�<@�R<��u��ʦ;Y�=�5=4k�<��C<��xNI���=_5|= ���5?�@Q<�~l=/�<	rh=^C3=���<�����<g=�m=�)/=�<�fK=PI#�|M�hg�<t�f���*>�����p>�꒽����휽c�O�k=�s�<�:��������@>Q��BܽL)��Ƚ�ӽ3W�= TS<�_[����k���q�%Q�=�3&<�o<���>����蜽hx����/�` 6�Tl�<(���@�=�A���h��\���g�������j�<$�=@�=���=��<�y��T��Ɍ< ks;�Q>�������E�����uB���>|8�L ڽ!G>$ｼ#7��0༴G�X\�'Ž�5�L�T�B��=_e��t,��]��� ��/�����5>k�A�g\������Ȕ�8��<JGE�7>��*<��T�����">卆�я��pǸ�ycؽ� > �; |n;��=(�%�02��Ȭ�<�#��`m��$��B�����}=D<>��_��]=<��<�AD�Z�>H����]=�����9=�u=]f>�=:� �@W�E���==�e�< �P��Pt>����e8�,ρ�G�&=s==<<��l����s<��������¼���;T���h��<x�ټh�<<`f��]�= �b<�[-�\y�=��{�٫7=�h=Xĺ<`�� ���k!� � �߹8��<@�8���==�u�<�O�< �_=f�=̰�<��3�:��=�T<����&pE=�v��u�<��=�,m<@��<��żP��<#�"=��m�h~�<P��;����|Aܼ�=f�=�=����x��|<s�`�$���E=$��<8�=�[[�g��Ξ?�А�<��=�p�<�W=�G=�36=���<6xs���>;`ʛ<����`�<�)�<�R= 4>�d�V���<<���= ���Ϟ=(W_� v#��|<TF��=zl=��T�(��<��l=�$x��]=�w�=˚'=� =`V�;�^�<��<�o��"�=\s=�x�:�3�X�� {;V��=�Mt=L���l���P�V�Wq=d��<v�=0�O=�+<�H�����=��G=6�=P<<�=�3�@l0�T4�<�>���W>���ڶ�=�7��lmx�������=���= �;��"�*1V�Ј>�Զ�����\B�/-�ރ���ż=������m���(� jĽ<{X��%N=��=�l{�<�{D>��t�e��|s����@��������ς=�᜽`���?�;�R�:P����%� ��pm�<�u�<�k=��D;�׀�ս���$<� A<��=|I�4���	��P�W���?����=��x�*����=>.�F�]��W�ߗ��|0��ٖ��y�@^���f<�=Wӽz����;��ف�0�»��>
���f:]�i���,s���+�����Զ=�}l=pf�;�
ؽ�2�=2.�N!1��+��Y���V�>��<l7�<�=@���6ꅽ<a�<
��~iX�D�`qR� ��c�!>f$a��HB=�9�����=$�������Ud����O�R=���=�&�����@ؼ�f��9���ݝ<s$C=�>p�/�P���ȅ�F�<P��<ଲ��߼�; ��:�N��@Q�@Y�<�!k<`g�<��%<@�Z<�"�;Gr#=`��<���V�9=x���<�.+=x�<�\X�F)=8�?���>����<�< :\<lo/=�(�<@_�̅l=��<xDh<�J�<~)=P	<�P�;���<ܻ�<p�;kz=���<��; ���D�<���<�����<䫻<�开_N�L��<l'�=����Hi��=,��%μ ����=RP= �������׼�ռ�9�<�ex=���<B�f=Z�;�e�<�H�<�V!���Ҽ�u<=pM�; ��;@(R�q-G= T>��Jʻ�^<"�&=�/<�s�=����/ּL9�6�h�a�7= _k:}~�=|��<@߹�XAV<�L�=��=���<��A<ha�<��
=`�)�,|���6�=�OJ<�XB������xV�8k
<�H�=vQ]=�dֻ茵��dz��>=��<0V=��<H4�H`I�n M= �
=��=��ѻ%=y%�@�U����<-�;��=��c<���<xz*���M�ov��9L���<�>U���c:Q�t��=@��V�l��CI�3Q۽��N�pH�=X<-�X�H��d��|6� i%�8�J<���	)=`s�=D���8��j蝽�����Ǻ� �G�yO5=z{�D����J�;L-��`���~�S�$]ڼ@��:�2[;�\g�P켼)S�� �����@;�K<�~@=Bm���e?����� _7�b%�3��=���eǽ�z>Y�Τ�� %�C7���B�H�����7��{e�L��={ӽdᒽШ
��멽��6<H��=����4��-�ѽ(�b��K�S㺽�=G��=  8fʢ��)\=� ��f���%ͼ�̔��� > �	=���<d��<�G ��,���;�����L���H�'�L���(�=6/��� =�x�S*�����=��������L �ֱ�Z	= 7;�U��K���>P�K���"���<H�=�X=D���0�ü���� �;@�ɻ���Z���q�p���h��8�g�\3�< ��9Xz<FJ=x
� ]��fK=���;-�� V�<'4<$Z�<��R= �/;��i��a-=�욼䓽��=��R<@���%+=8<x)g�EPP=�.I� r`���<`R< :o:`�;`��;T��<x���D1=�&= �����;�s<�Jv<<ԼPz�;ܼ����>���K�<��z=��|���%e<		� >�9dҳ<�<0����`���Q��[���Qi;�Z=T�<��\=�����
����;:i�ں���'= ��H�� ���,v=H�c���
���<HsP<P�<�
9=:׼^T��{��C�����<���;�|�=pP�; su�h�
��q�=P`	=��û�8�:�G=,v�<��z�D�׼C%�=�0���J"��:��.� P9��>'*W=��<�t�꼔䨼,Z=ؾ/<�Q=̚�<\ʼ`W�"�=ȿ#<C�=�����(�<*
dtype0*'
_output_shapes
:�
�
Const_3Const*��
value��B���"��-�=4��<�k =f��� _�;��U�,_�=��ټܳc=!<��6;��C��(�*�z��%��V���
�X�K=p-<���H�(��=x$�$ڕ<����N��h�= ���V� ��;\d=п���ü���<Rd=��<ܑ�<�S=�*�=�Y���=�d�� y�;ˎ=�M��F�;���<�+<�U[���;���<N�&=4���sv�=��<̛�<�9;=�nM;��=2Ň�%�<=X
e=�S=�b=���H��Ƞ��p��t��]����m�:��;�S��ʉ=�`E=������T�?=`�&;��o�s`���=3= :#< �7:��1��t=�ݶ�t�O��м�N=�h�T��<���=�J�1ߐ��|�$<��`�;�Or��ꕽh1M��һ
�V���t=�J=���:�����T=�'�<w�� �=�9ļK��=���<.s=���϶�=���;�֟<{��=�&�兼��=�Kz��=4m��d5=�=.;��%� 7���m]���R��$c<p��l��<@�"���]�0�3�0���з����X���"�8��< ��9���� 4�� �~;@��: ��� $����߼@�:�=ȓ�����LK���x��d�����<(���w����� �����=�
8� ����V��01<��5�R�N�h*2� �g�h=��w!���q��3��@�;:�<<ߐ<<G��U=Z�w���=�Ug�x@�<�:�:�1Ҽ��������;8I�(��<��&=t�������s���c��TT�`�����<��A�L��< A���r�p,x<��X�Z��`= ��n%��ű�D��d>μ�a�.���9\�����X���@��� ��M<0��;�mn�@���v����)�l���I����;x�<H�z���� �ǻ`�����e= :]���F=(J���ջ@[� �`�Pʄ;�0Q��<tb���@az�@���z�; ���@�>��"=@ڴ�g�n=X��<���<�@����9;�tֽ ��<0�'<� l=8c =�*�����P�K����,��.�,�μD_�<��=����y�@θ�%�=P���c�%=��Q�Pe3=p�A=�<r� �$g�<�Y�<PW�<�d�� �r<�fʺțM<��={�Z=��=$�<�V~=���<���<�E�����<�nJ<�+�=4�P=�˄=�!:=�m�<C4N=�=�Gp<��='A0=4��=�|��5�Ư =���<�֥=����� Bu<<��<�X��9��ʶ<�l���<��M=(�<Xfk<f�x�4"2=h:p< �d���� Ѽ �ڻP�v<x�v<�u�<��u������-�X\l<��ѽ#;�=�]3=L��p�H�r�=�����"R=���<�{�(x����<�]ż�������<�=x���%�<v��=�'{���I=��=���.�E=�E�=L=P�ệ�x=�9����V=���=L�<����(�)<��9���`=0O�of�=K�H=@d׼��B�H�R����? =�PW�n/
=���;:q��;�,%��H�'�~�w���x�u�X�=��<��C�v�/��jɻ��<�q1=ļ��2�� �G��!=<�#� ��;R�7�^L0��u漧2d=|�'���i�ЭR� /��ѼG����q�l˼$�#����u	��p��;��^����r���������)�: y<d�<)��T�=�0���:=n�J�/==���<B�Q��b�`Z����O<�fP<P=l�h=D$����2<hv<T_f��E�;XcF���<����;p9������<��i��k.: 8�<�m�@�<� �y��d����Ҽ@!�;!/�P��<�O<�����H=<� ��K�;p�< -��v�=����`��+<��k<,���I�<V���h��`j�;�j��ܩ=P׵��<���@��<𳋼��?��j=$4Q� 7z���;(+����4n�<�ܥ<�� Y�XRc=��)�͊i=�W8<�*X�xP���j����E�'=�c=D#t= �:r)/����� ��<@q<u)���v���=���<'I���#����S�b= �ݼ䮑=́���=����ȇ<��
�8�L����<ʝ�=��b� =���P����H=�L=���=���="�=h9��Ј<�񭺐Ď�p�̼ i����S
>�)�=eC�=�z2=��=@�:#h)= �:�傻��=�o�=��5�l��� 6B;�UC�r��=>g)��o	��k6=̐=
�<ؼ��:=wؚ���w= )�; ����;=:v0��+\=�h<�'�=_v�_�������7�<�,=,+�(�q�U�<Ċ\�06A�� ���=0�e��f��G;'=Y�=���<ׯ�=��n=��Y�����*�=�J���tպ�n�;�"�<�|C���=fc=��R��+�=82]=L��x�Gh�=��߼��; �B<n�I��S�=)uL=:�V=@�2� �%��i<X�<���t>o5�=dB��.�� 3M��qr��o=lO�R�=�H�<��n����0?׼� 9��%���=����<���=�q"=��"����� _;�6e=�+�=J����̽��h<��U=��/��=o@���
�x�b�=<Y���A�8Zr�@����.�����<@ؼ�������J.�mA�B/=�_#���w�8�G�?�<��X� �y�(I<�U�<��l��>�5��=�K����y=�=�oe�  _�l6Ӽ=�M=7�\=`��<�6i=�k¼�DY=��d=�A����.=��W�h�=Jsc�x 
<�
� 
V�l��<ϣ<;�=P��<�%`�p>�;��7=�Cr;�-���p=��D�G��=�Zh<�Wڻ���<����T`�<�!=Tؕ<X��=��S�"9=��q�6n=f�5� �6��/��� ��<�Ἦ�=���� `F�l����2=��ü`λ�"Z=R ���
�hm�=�o=� =/�1=ҙU=P�`���r<�M�=���� �=���d?������dvK�ơ������=���<{��=����^�~8o�{n&=P�+=��彐u�;$C�<`��C� ��;>!��4<L�2�vE�=�.��ft�=
��� l<��ҼJ`:���g<7��=H�O�n==²1�����J�<�=��=��>i��=�:��8�n<�����	��dⰼ����(@���>�	�=���=��=Q��=Ԕ��U<<ඊ��jQ��p�=�3=�T�;�V��()��D6��[��=fQ[�TxJ��@'=��=�l=�?����(=�ʽF�E=c�\tb�P�=�e��&^=�;ә�=�����|GI��F�<��<~���1W< �J=���L(���Z@� ��=:jM�|�����=*֕=�u%=���=%[k=�M�,�*�S�=��P�0!�<�h��G#<@ST�>뻕�A=Ǹɽ���=G�=��*d�"
>V+��?Ҽռ��Ž�q=H��wFE=l��0Z��>�=Э�������>NX�=X�K��:���]��H��S�h=�$��<l��<�]�7�׽t�����dĽ0�R���=��=�=p����������<�A�=���=t׼�dϽ ��<{�V=�h�i�b=ij��q5���(�X��=���������B���������3K���`1��b0��a۽��;�н�r=h���&� �q;T�<Vs��;���\�;��I<Gӊ�tF">@��2�<k@ս�\=�=��<��[�,"��x�=���=�l<��=0|d��6�=(�=넽�|=O=�?�<�V�����< �	��ns��D�<��=���=���<�i���E�<6'�=���<`�����3=�WR�h/�=���<�$]<���< ���t��<��6=?�=m��=������K=��S��6^=~�1�0X��X0��0K<8 �<�3�糓=�̧�x�r�\����g=�����K9�rs]=�h��f�3� >�Z�=[L=yoD=P&�=Р���=�ё=�~���K�=�]K����G���Ϙ�*W������]=��;�e�=@ވ����K����=��C=���x��<P͆<��ӻc!̽|]�<_�x����8�,ҭ=Rͷ��`�=�����=�;��]��f�����<��=$�h�GV=�O^�H��@x�;��,=���=J��=[U�=���� �9�����@��b!��}�fY=�E�=���=��<��>�T����D��hI���gf=X��<x�&�� y� =�����
�<��\�|�C��=/-�=h}�<�f���Q�<��Ž0��;�g�������;_����%=x�x��ɴ=��;��ٽpD��4�; ��:��̽���:X�M=N�ͽj�V�`3�kE\=�$� �[�R�=P+=Ⱦ=DQ�=ɹ= �����r�s��=���M�;=G����J��|���&���<!���=���<h�I�����d�>�d��׻��^�%�����)I=�qj�< =5߀�����Q�;=�V� �8�mQ�=�L�=��$��P��P���H�<#=$��� �-� �<����i��<�ּ`�R�Gڔ�@�����=�|�=�z{< ���j��+y<�N}=]ߥ=��;�|�H�><�O�<Ȥ���&2=�gL���|���w�=`�����(���?���3��3������ ;�P���1�M����N]=�=�����J	<��<z����@�h�m< }<��X����=��8<,>�<g�����=�g�<�:����;�c�6�h=�K=�Z< �<Pn�;ie=�$=�E���=�m��i�<�s/�`~�<X旼X���"<<8�<�Ka=��N<���@�;
�=���< �=�@�<4���C�==\-�<�z�<��x<�/}:P�;iO6=���<�x�=P�7<�~=��m���L=�V	���3��ӻ0Ԃ;@H�;`���/" =(�'��.�@A��dnF=�̫;�|7;f[P=��l;�{��<�=�9p=��'=@��<��"= ��8�=�[3=�te�;��=�����E�D�x۽�i�0�� ��<��<{�y=��U����Ľ }�<�=�������<�9&<P��;�>����<l�輨�{��$��A=tv����=v����<1�ϼ���XY=*©=�B)�7��=5F)=<���8�����_<�؎=�V�=B�'=8�ۼ�K�� �m�
ڼ��C;�[6�$n���¼";=/��=�vV�LM�=^ڜ�.꼔��>���7=���<��m�(�T��������ବ�b ;����8j=⧛=W�<hV���?G<�ۗ���P���@�R;�ļ�oW����<��D�z�[=|�<��|�X�}� �2��]���j�l���@�+<2���������xN�<h$��h�1<p黻@򁼈�C<g#=ȸ7<>�<L�����=��;���= C���[�謧�`�����ݻ��˽�.<=�e���B���j�?��=8\��5���~}������=�H�����<oK��fF����3=����X3<}�T=�<�5,���$�	��/3�d9�<�N�0�L��Ŷ;���IO���Ҽ(�3��ta�b�6�s=-�P=� ��@C���愽��8�<i�<tj=p5�;���������'�з���i<��%�_Y�W���3= ��9dH���3C��ڥ:H"�&��L�ԼQ)<�b�<��6.�˕"=p�l�@����6��L=�$㞼X�N��� <PN�;�[�r~�=Ծ�<���<�۱��/<8�f<6�t�h��<�	����<�ǆ<�r�@`�`r�;w#= �~; ���`Y[;�ץ���=.�� 5ں�Z�P0�� ^��0�Ȼ%zQ= ��9l9@�*̻Z9Q=��U<�0׻����n���	�<�$<<���<�I�: �չ47���6W<:B<�j�=0w-<@�6;@g�;	=��������绀�0�d޼��/����<���@�O��k;n�=H�<��ܼhS=0�<�0RV=�t6=�<h��σ;������J=��j<�������<�a<q}��諾$SR� b��Z��=*<��r��=|i=��	�f=5�@�S��V��O���z��۫<��t=	��= �:�zt��擼���<�r���Y= 5}���*=Pzu<�H�<�3�D5�Vp�=p�a���,�#a=nŒ=��A=V=�2�D��=�3���= ?���;g�;=������ܼ4��<����Z��l�=�o�= Ka�6�X��<��<p�N<�k�<�M�<��>��QM)=�&=�哼�-�=d,�� �ƺ��X<<L�<�P�<�pG�4�����μ ��:i�=�\�=��������=4'���$��#/=�<�F<4�ἈH���νp�p��c �@ ���K��r�<�XW=�f8������X��T8�h��<؃{�(���ʽ觞<�,B�XL$<pc�=��;4F����=$�<��"<`�=<u�<(JW��g�=��{=��;p�Լ(�M<Ф�;��=Ӏ4=̍C=�99��wX=�7A���<,\���N=�e$���ۻ O�;�Xz�^3��s"=@Vλh�Z<�����f?�f+�07� d�;2�5�(�ɼ�w���<�#ȼ���:������<и<���<�Mż� ��psP��'=���\�ؼj��0&���曼^D"=`��`*ü��C�0K�;�C�:��ټhI<��h���V���,r< ����P��J�W� %���.�в��0�;,#<.��x��=G�Β=r����<`�]<QB�\ꤼ@��PR�<@k; �M<���<d�뼈{�<`v�<������@nM�t��<�K���z�@�@*�:��<󃻤�<h��<�����< B��L��� f�������9G� �;���� I�;���:<9�왃<�p���y:�S�<Bu��<�ۼd۳� �.����:�jO:x<����b;p�0l���.=�f����2=`Լ��<�r`�J
����ż�����f�;�䡺�ٺ���<x\<�����S=V2=��g��.&=��;�wͼ��w��|�h�_�\ �<|�C�lQ�=�l=�2��������D< �+����� �<wĘ=��3=�X�8\���7s��9=�Q���l�=��Q���I=XaC<�a�<��!�`�;���=[�<0���HE�<$�<O=��=�Uq�e��= g��S�=8����ջ�!<f�����2�<��弘<�$�=��=�[ʻ��D���<|��<��;�<�"Ї=Vn>rS�����`�M<���3v�=P�Ѽ Q�:�g3<bT#=��=V������k��Is=`�<˪=�����Nc�ȉ=���� ��9F\�̗��؅p<l�<��^:�q����D%ּ��޼(;��餽K�?=(�<<�-» M���8����}=���<�������{5=^)D��@<�+?=�!=�ߔ��\�=�|�< �]��!l=��5=Ly^�A�O=��=0�8�|��<���<�H���-a=��-=�0�=,+��`�}; �&; W�;|g�N؀=���<|¼ ��P���U�1~v=��O��N]< ]a;־A�N�t�$3�� ��;���� �@��:(|�=��|�����`���A< HT<]=v��r 5�p}e��%=`u�� c4�^.�x�-�X�ۼ1�Y=�����*��E��XU#<؊��RdS�Ğ�`��;H�3�T���nhR��׼<�(λ����U��|���&�H<*� Ψ���;b�D���=8���DՏ<y����]I=X-=����p &��	l��̵<���< ��;h��<���AI*=y�=t�X�@�!; D�h�<����H~��(���±�'	=@�g�R�=�6?<L���|ƻ<`��;X�W���;xs�~w�xʷ<�΁�`D4;�)<,�ü@��;@��: �]:0�;=LU� "л�<��x�8<,㹼 q,:��&�1F<`��Ȑ���qE=��(��="����=�..�����><�<����̼�	�<�$�;��D;
=$)�<�^��Sh=��4= �ߺ�{'=�<���3�(�]J��_��l�6�H-l�=�L=�:���*�m�����< R�<�ݽr+&=��=���<���U������z�<�ʟ��S�=a��~�s=����߃:J�4��4��"�L=r}=8F�xo8=�Bٺn�=;�2=�?����=��<>�=<�
��6�������\�v$N��"<d�I�C�=g��=���=��λ W�<����H��<@م:��;��b�=���=�i?���t�p܋;D������=ߢ��Hݻ�V�<o�=�ej=��ڼ��;�o��K�= /;x� <�к��S�"�=����.�=v^�E`��PS5<�"d< ?C;�mJ�\
�L������z���ǽ�̈=M˼|����;=�9}<���s1�=�m=��� ȗ��h�=�]?�\��<p��<�2=8=¼((�=̷�<rWE����=,�=�M��������=4u8�p<1=@.�"�>��5c=�ξ<���=�����*��,�<�Ĕ�Ƚ�4��=p@T=8�ͼ�oG���D�����c�=t>���}<��b<���-����)ż�$ <%����:&�,I�<\�=p�������a8���/�<P��<^��=�&%�j�}�P����=�f;@O.<�?���Ѻ������=��إ)��O��@�D;8�����n�VG��ş:����ƚ���o���=��� wc�f� � X�8:z@�<⑼�����&�.�S�A> �k�@S�;=?���u=)�=��o���;�PT�C�0=<�!= �ڻ��<l�켍s|=N�c=�v��|�<��鼈-X<�����	�����x-W���=�k�; -e=�k<���<'=�/ =�HK;ؗ<�A
;jso���=�k-�`R&<�n�<��h<h<XQ<���=���h�F<���0��<����8�^�0(\�p<�<�(;�
���H=�4��˔<DM���F=�$��R��͜<�˹����2�=p��<蓈<�%0=��=t,ż�}=�G:=��M�W�Z=@I�6����D�f�J˫�������< N;\�=,z��%U���Ȁ�"�	=Ц�<���
=^�=�<�(ܽP�X���)����:(yż��=(@���ч=����Ǽ*W��I� 3S<(�=�Y�Uf`=��j� ��;���<Pis<㭄=�J\=��=H�d�Xu
�o�� ^��;0��Ő�\4�����<!��=��=���:f�=@=��䍃�`?��x���=��,=0�� ����4���Gu���=�,�������<s��=r�d=�:���D��޽�&�=t?��Hꄼ������w�1=�5����<=r�:�g�Խ��v���;�輼�V��U��P�S�>%��@T������uT==;伌d��_�=�F�<�"_<��_=��R=���v兽��=�ؘ����<p��;<��<��ռ��j;�=����@�=t.=�]�W����=�N�P)�<F�S���=L
�Ds�=��o�f�WmN=Їμ������=�|=앮��8I���o�-V��,�w=4�ټ�7��2�<�����ȽPj �H,<y=��8�2���U=�.[=�������;e��@�
=�$=/6�=���z�w���޺���<�!<�g�<:X����<���U��=Xd�x���T���B�X�� a��>�@��;ܲ��^� ��vE��&3=��8�Мﻌ���P,�;^�[��}�ݼH���Mg��s>�!�<�ሻ�Yҽ0]`=(��<�>6����;�����v=S�L=�v� �;�p��nę=�x�=�A]��i =dxϼ kG;'k��x
��Xe���햼�^�<���<�{�=P�;��1�5f0=�>=�2�<P�Q<`iT;�ŀ��= _���:�<TK�<����3<xv;<�Q�<K��=���l��<���)=���8v�� �@�<��< ��;d�V����<�����ȫ;x/Z<��b= 
ݻt2���N<@w$<j[�E�=
i=tL�< ^$= ;=�U��X�={)=�/�g�=��̼��ؒ{�r���Dh�����%=����w�<Ŀ鼕���5)Ͻ���<@5<Hh�� = ��<�<�:3���0i�lQ��>��T��e�=h��0�=����%:���������`x�;���=�����M= Lٹ8�y�@��:BO	=�JT=�vd=���=v�����R�50���H���޼���͹��g��~��=)��=�p�:���=��9�hZ���o�,����Y�=�q5�8}\�H���Dӊ��w��R1=8���&k�P#�<t�=�{4=���P��u���Z=`j���ƚ�0᧼I����=��(��o:=�=ּ�d��0��@�A��`��[��E׻���p�� /��t�O< �=��_:1�t= K	;0�{<��<J]=H7<X������=�����>=�Π����H��X�9���<k�۽�l==��\�=뛽���=�6�#��Z�>�����p�<�y���c=xؽ"���#ؔ=4�E�PҜ���=�>
=��ȼl���:R��ۋ��8#= '����ڼ��<�������r߼ �%<�?������[=4=����%�;J�a��b�<�=&��=h�����H@\���<�S�<HN�<��"�p��;ȁ���4=�zn��?ؼ�t�P������r�(��&����;��;��$����[�=�c �@���μ@@��>J9����L򋼈嗼��;�v�=�f=��:Ay���
/= i�<��;�@��;�0\� M"=��=nf&��a��R5�GkV=��=d�?p<�Ԣ���������0~޼L����i*� ��9��+i=������� �=@J�<�׍<��)<����q���o<���w�< V-9��:(�h_m<hE<!�=�o9<�Ė<�_�|b=���˼�6�`v�<��S���I� �; J�@"�; �<C=��;��8��_.<(�b<�ϼ�A�=��6=��<ч<�_�<���"Tv=�ˮ<pM���}=�ˊ�;��<ӽ ���P���x�2=��^< �=j����ݽ�<��L,�<@�T�d��<�Ã< %�;��H� �<��q��=T��΋�/,=�� ���=�p�����k��F����<�=�I���==DD�<�����}���Ŝ<[�U=���=,Ԓ=�۽ �c�H�	�"F-��$ �da��cֽ�\l�/+�=0*�=������=�2��T^���� ��*c<=�¼��B���.�����!�@Ao;�L����=ܼ�=Y25=��*��-���཰�<`"��������A%��l*�<\v��j�,=@#��`̽ �A�����\x�~&���������2�H���佈�D� ��;8��<`��;����c�:��{; �|<Ж=����w��=}���"eG=�f���t����t�>���<�M�V�'= CB����;���8�=�C�1H��ύ����.C��W� ��<L���:��l�=�M�������
=��(�R���P��d@�j|�4ӵ<�f����ϼ�҅;�F���q�l�� ��:���n�x\=�<>Qa��v��J�H� 񆻨"d<,VN=P������h��h���P��<@T<�f� ��9���p��< �;���8샼��a;����4	�B�����;�#����D��B��Z=�^�`�;���x�n�$B���9;�`���TƗ�T�Aa=��= =/�9����=�s�<��c��g$<�N�`�p< ��;Np�p/��&=��!= ��9�ڼ�@�5��z����ݻb�h�:G�����7�H�V�P���H7\=�5�����dc�<�-�<з�;�!;�F$�s2�����;0�ϼ 6y<�����;�f3�`>�� �ͷ/��=�y�<p5��_<p��<R�H���������F<z$�` � '>� {;�1�l��<��.=8L�<l�0�	<�t�<���GP]=�|= �-��]'�`�(�����2�=�*:�� ����h�ռ����,c�dw{��J=���=�z��%��=�T�=0ƻ�N���r�<��;�F�S���v�:=���=È�=�<e<�#�t$���?��0"&��e=�D�;[�"=�"�JJ-=���:��=4JJ��U;=�nZ<�>=�Y�=��=�ȶ��y!=V���fz<0y�;�2���=v_���K��G=�Ѽ�:ƽW!�=�ҝ=�3�;Խ�V��tHQ=�*j� 	�:���<Vk">�	���{<���<�㕽}5�=�Rl�`��;x�<�P<���=�B�����[���i=@l<gV�=h6��F�m����;ĊJ�d�ռ�O�*��=O�<#5S=��;�"������B�@�<,��/����u��=<��h;�t@�7/��1-�� *�<�#��@1E�ڿ���=QЁ�8�<�
�=2k� *w9�*�=4��<Զ=����<8���r��d��=�.A=�K����7<�I��Jc=��<rO=��i=F����==��u���$; ��Ȁ�<�r� !�9�{�<��B�>�M�r<X=�񆺠";�N�����ZO��[-���<0����P���%,<&�@���<6�.�>{=P1���=����,^�<U��t= ߾<���0U�����<�|��} =��k��q;P?�; �<�^R<(FW����Q\=Pe�� �0��A<x�=�D< <�9N�e� �ɻzm�D����>���+�������= ��< �{<*>1��A�<@�v<��=����;+.<؜=��<�}ȼp��B~#���F=*�=�%	���܏�<�%�<!Ӕ��������;$�<�=����y�=�������;A�i=@�_< Ղ9_62=T毼b�s��ގ;ĥ���r�< ��:L�ܼ�$�;����3g<��	=Ps�t����t��fC<|M�<������ռ��=�_�p�ƼX_<���(�.= ��<��< W�:^�Z��9����<����̧�<��<.<<�!$=4��<x�Y�>��=�h�<P8���Ƽt//��X���0P�`��n=b����%=�#�=0�ӻ�54���;�n�< u�9?��N�@=��=k�C=�_��54�`z��0R<�ʼX��=��'��J<�����=.���mT=R�=8ӛ�F�= *�;�ߑ�J/�=���<�yc�`M5<�(�����< �����@����q���!P�,T=�@ż0�߼�>M�q=5��Ƒb�pj<@=����ȵ��@P=iz�=:����t�����:����v>6������ �6�@̯�E��=���������#��=�k<b�L=8l,��6!���;Xt|����#o���r<h�v<s==H�!<Y�N��p&��0ʼت#<n(]���<��̼��L;���<ZM9�b�D�r_:=lt�<�p���l��x�=�都XL-<�(2=��=ؿt<:�=���<���<h"<L��<����E+=| =�-W���=`�;�/<(�V<̃�<-o{=��]O<�维�<�9м@˦:(Em�X>��P<dlʼ��a�{_= E�;P��;@�ͺa��8�������5� $\�T��@�*�K�=�(��c�<���Ѯ<��M�P��<�y��� T<H���$O�<�A=|��K��ڄ<�a���Z�<P;Ȼ�W��h�<���<Ŗ<H����J���=$����x���܅<4N�<��6< x}�b)N��\廾�%�<K��B��U��� � ��=R=��(;��)��\ =�x�<��6���<�<hN�<��<����8x��,�hY=�<����p`��$@�<Ҏ<�m��Ғ4���:<�ˤ<��=�߻p_= ����-�<3K8=�Iz<��u��cp=�!�r�R��u�;>��p�<0>�0����:�h���z><DT=��� �9�{�;��)<��< ��9l�	�� �<x�"�������;`���m�[=���< f
=��5<�F�(&���]8=�&��l�=p)<0=�;4I�<���<X��[Q�=��i<�h�;0�Y�`�I�d.弼����"%��A����<�Z;�F���v=g�������Ǽt �<`�!ĉ�H)D=�=��<ۘ��P�^�0��;�^<0O��!ͭ=LE������D�; ������Ȟb=�.{<P�:< $0;���<�G�hHh=��;pŽ��»�?��=p��3S���6vR���e�?	=��� �q��p�=�=����0℻���<|N�<�#���0�V�a=[+Q=�����F��$S��1<���=�;]������=��b/�PY=��Hб�1��{�= ��;x��<X��C�� <p��;�a�EG��t��`��;�h�<�*��d�<�
��C_��f,�a�B=/d����<����KĻp�=�4��xe���� =��=}�����4��<��W����;�Ȅ<�"�=L��<R<=\�<�|��8<�$=H��|#����><�zS����= QA���"���� X�9[D=pż�@��j<Py�<t��@�<��m��h�`�<x����m�_c?=PN�;P��;���� ���� �����: 0q��7�3<|��<�90�'
=�Լ�<@»\)�<@_d�`��;PVN�|��<��#=ب0��d����<@�x�<�x�@���0�'<4��<���<�}��=�S=c��x��<X��<H5#<�����@z���*������мЌ��D,ռ�b�=Yh9=��b�/�4!=(x<�f����<�'�<�Y=|x�<�O���y��|��*+a=���< p��PG<t�<�� <����RP"��rb<�K<�ư<@� ;�=u=@9���V[<g�7=�m�<`��;Tjo=(��(��'K;��`�<��9� ���i��h�L�$.�<�No=`e@; �< �~:'J<P=<ЙĻ@$�&�	=���Ȃo�P��P�»��c=5V=� =0݄<���ͼ�lU=H͐�W4Z=��<��;4��<��< �h*�=0�;@W?;���:\rG�����@�����;@����m����9:8��`y?�d���,�ܼ_6���ި<���旽_|=0;A<�R7������3e��|�; ĝ;4ɼ?�~=,ր�*8$�hsG<��B��������<؃#�l+�<���P`�<򼐸�<8ԥ��a<�Z���L����T=Po��������ʚp��-�@��:@#���$x���=��=�H��UK=x�<`-� ����J���b(=N�,��� <�x�0�ٻ��x=�̚=\>����u�p������(��<�S����[I��ȹ= DJ9 &ɺX
������Y<�M<��3����,���-���H�:������=�&_�4��x�^�?j=����X W�Ğ�����c:�=(���<�s�<L;�< ��ı8���{<�,���t;���2K=�F�<�i'�0��<R��L�:�t0=@�!��X{��o�l�1��6�=4P���{��F�t�\��V�<��x�����Ie=(�1<�^M���<8�D��17���#<��P�1���=�L���%�����k��zBh�JG��N�: Q�;@]$�,��<�)��T��jT=t젼B�==��<���<�7�� p�8 ~!�P�o<�=�~�;�Q��%6=�,L�`�<^"<���:��; Ay�,��<87��R�,����<P�%�0��J�=�\�<�ي;\򍼞R
� �����;��A������
�B�����=�8W=�мB�v�Ф�<�W��`RŻ���<t��<}[F=,1�<�霽�v� �i�ٟq=��<h�4����<4�< �/��������u<@�� t�9�p�;a%�=��Z;���9�X=@�<�.E<�%R=Vm��>�pn��27��ٻ<$���w���K~�(r���l�<�=�j�<h��<0'���a<�<H���d�8�= �%�(��`Y��D����o.=5Ė=ln=�-^<���� ���O]=�O�f�=�+=�G;xCH<���<`'_�W:�=���;�^��ȋS<d�K�Ȅ�<@=�; )*�4��<�gO���=*#��w�t0����o�����7�<����I���y�<@{b���Y������G� �� �_�l���/=42��ʼ@���wi��`�M�D뼔^k�z�=�n���<�D��'r��Hy�<Dwʼ�`��r��=.������Pr��:�t�������l���n=�Lc�<��{=��n� �=��Z;�틽������P�=SC̽dn�<�G}�����V�=(�/=@㟼&����՛����$��<�o��R7������=�b�(�a��Q��M$��X��<���0򼏞���޽��4�lf޼U�ý(�=�� <h��� Aw�V3=�¥�Ω����;��;9¨=��a�T+�<`�N�� �<��J�d/��	�<��½@/};�����jE< ��;�$�����<����p� �P�U= ����3Խ��0��'&���<JT8��ǒ��D��`׽���;K?ངo2�dR�=��Լ��5�8X< ݼ<6����_�ZSr���< �`�DO��@�z�@�!;j�� �8�@��;�<����O�<WH�ψ���H=@�����<�<P7^<��;��`<������
"=�X�<(f[�Db�<�q� �(�hJ<D�����|�@%�;�]�<�qԼ+��0
<�&<lU�@�<\��< -�`I��nڼ ��>��PJ+�0��޷9��/̼�=�r6=�����a�0]�<��E;p������; �}<0��<h�<a-���}�@����=���;���;�����;D�˼O؄��.L��JC<�.�J������d9= �߻P�M�$�=8v?����;'Y=f J��9T��i�2��o<\ʼ ��;(C���P�8�<��=(ھ<��;pK�;���<�৺�� �G����<$ �� �:�$�� %g����<� �=T�=�W�<Hc���4#=����}n=��<&�0˻���; -h;>{�= h���[��vI<p'd��B�<�k<�C�a�E=hԢ���=X.��~˻����>��T9��
�<jq��i��,��<��1<8�0��U� $.�����,c����ɛ<��7:��<�.$��˽ 9�ɍν�|��A=R0��@��;0��;8���,$�<��=�l~�<`N�=���� !y�Tt���+�:�`�R޽�fz� |i;Y��=�i�έ=���W��lf��f(����<��ӽ0k �`h�h�<�ը<���<ܞ�f8ԽH��<@[o�0��<�E'���)�ֽ�W�=��}���u������hi<�F\� .92�[����H�)�.�:�y�(-<�u8;zm��Aw��k�%N���UF��9J<�=6G;=�����#<���p�<�y1<�+]��w=S��8"-<�ʲ��� �������=�<#�ɽ�Vj�h��<���<$V�X�x<,fi���9�����٣���J[�u���ɻl���9g�n��=�}����ü@��;�Bܼ��B� �¼Ȕ����r����;�j"���� �x��C,<�X<��B�� ��; ͙��+��=��'�g���-<P�'��ȟ;H�M<Tɡ< �; �?��0���6�N0=�<�PǼ���<Pjռ��s�T~�<$��(n�� 9 ҹD���$���8����;2D?�P]<���<`�ջ0��;V��(�O��y� <Z�4o�>b>�,�����=��=Zj����7�=��<��@�; K���.+<��=�)/ϽL��`?����*=_���;䍌� xt��$��OW���H��8�X��*"�0X� ��K=H1��޹����<P�
� �9H��<>c����\�м��y�P�;�y���,<<0�2�����Hj�K?�=�7�< #���H	<h��<�s׼\������+�<,��8=�>��W�:��ܺ�h�=��+=P��<��Ǽ�8ռ@P=�ж�|nt==�<<ܦ�h�Ӽ�<r��P�<2�=����g��@����J����* �`�,��o�<F&�=� ����=59�=����IX<��H= � X�:��C��J�<���=R�=�:�<H7g�Px5��;L'�����<�J|;pӁ;�;��D\=�*���td=��=^���`5�= >z� ������=��=Tm���O��ս̞�P��<�帻��<@C�:v�y���=���<rҜ����=�b0<$�	�X���<�8�=ĭ���
�:h�����>����f�;���<�±�PH >�!ɽ�m��hu(��/��ֲ=f�P� �9:2=�v=���<ɇ�=$#���L�<�����4�dz>��O�y0�= a�;�ѐ=x�'=p	�<}y� C-�Ү< z<p[���|�𽈼����ߑ�ͽ���{�� fJ; �|�@�׻�ή����<f+c�ԡ���A= �@�(�<��>�=⮄=����c߼ē�w�)=��<®����=8�`�� �= ����p=hQ< ��;��n=H��J1=��o< �"��l���E3<�<�<�
/�E����]=p�󻀟�: w��1J�#y��@���Ђ���F���6E� �к��t�>*g�m�w=�hh��CK=p����y=�pټ��<@���+�<���<����Tb��v�=�v����=$�<$M�< <�6<�#=�/m;r�>�j?=����`���Q=��=(��<����ʬE� �a��8�P����b�������6����=$��=@F˼��D� ԅ<���ʴ!� ��<��s<<�$=D��<^�R��&�����=�4*=t�����<�=0��<���.6���<�ݣ<�9#= �;���=&]� �����t=T�6=�
�;���=>"�/刽�wA;
��t,= h޹R׼`�s<0���P=��O=���`c�;@��:d$�<�F�<����ɼ*s=��; �� ȧ��� �=��6=�7==���t�hB2����=��Pj=�u+=Wu<` =�>=���;:��=`&Z< bz�t�� %d� �:�?	�У�;�$2����=p��� \�8	��=p/��P�=�-�<`��;`p��X�M�\��<�Μ=�=�	.����X�;��<x��ҙG=@�.�Τ0����;�n=<���h*>0�<^ٍ�;�X=��U��y�s��=`�;�D3/�w�\�Խ\Ӕ����;���@���h��j�X�^ү=��<x�&�tC�=�9��
B���ӓ=�(�= ?x�pN�; ���*�=|]z�����a����[�>Ͻ�;!�2�Ih����u=b�2� {�;��%=�/i=z=��<=4#ڼ(`���K��|�;N����{��F�2=@A�6��=�=쌠=����<�>�,���ƥ�=V2$��^Ǽ���PTV���T=:mO�X����<�L�<�߼P{� s��J�?�֙M�X7@<��e=��`=��=�.=+In=ZJ5���<����/=�2���2�E�=�Z�<�F{=�)A�`=�F�;�<�=�a�H�=P���)��|μ�H<��)=č��}��p�C=�; ����/��R����ʼHz<�����E<�'0� �:�#�<V�H�}ք=N�3���.=@�B�|��<��s��@J<X�r�즜<�)�< r��I4:�v�<��8\�<��<D�<(��<�ݶ<�BV=@�㺤���W.V=(�i����;	�v=X�<�Vk<�-���k�E <Z�3��W� 1����e!�Յ�=���=�������;����d�߼��:=�3�<�%=��<}6��G#�X�V�-�=ϵ<�w�;�[�<�lU=,�<ׇ������!=���< �<�};<f�t=��L� �R�R�=)vk=�<�i�=�:�^B#� �z�j���9=亜� ��䩃<�_����=���=�:-<x�=�X-<`��<��<8���Ǽ&a=P�;`a��0o�������d=I�m=�7=`��:�E���Hl�= ,:�ph=\��<П<�e<�u=�A�<GV�= J�pM�<�Sb�X�ȍ <���xHi=�:���&�=<)\������W<�(b��(=�3��`�<��-��Ψ<���<$:�< FS�:���½�,=���<`��9&=r�����P�3= p�8����m&>�?4�<lL�,��<@��:���̋�=$|1�Ϲ;)Ҍ��4ʽ0���8@t��T�:dDȼ"�*�݇�=�J�<��M<���=�瞽���;�餼���=��<�΀��a4�@WڻRB�8�&<�ݎ�����=#��=�½��z��.F�%ٽ�a<����a&�4 �<�Y=���<P�{<`E��X̥��o��G`=dD����ѽ@]q;|��^+=n໼)>��ۼtz��|�Z�i�>�4g�Z��F��¼���=��)��ڼ�y�<���<��7�T��$�2�zI���G��&���[�=v �=X���+3=!"=!쐽Cir=()1�0֏;>Խ�����=�J=.�=qh�� ��:�.$���;�h< 4Z�9��=�ZT��դ;�և� =�<��B=��+�����er=��D�`�ǻ8����Ƽ��
���<�f��x,�<��9��@�< *�&vw��t�=&�4�=�w�;p�~< 㥻P� ��d:�<_�<��h<  L�n�= xM:9*=��J=O==BQ=�p����=�� �^�	���h=�7����<�g�=���<�,�;L�� n�H�<�b�(�b��z���:�vJM�rZ�=C�=�[Y�^����,)S� ��:jxD=b�?=��j=4:�<
x�L]����;�C�= 5=�ø<0#=�Sz=�ϱ<����h-���C=�X(;(U<D�=�l�= �P;L���� =̱�=�a<�:�=����V���t��ݼ�6C= uͼ��U���?=�x`��e=O��=8F�<�_D=@ע�l[�<��=��� 4Լo4T=(�]<�懼�����O�Ȩ~=$��=VG!=��ѻb�#�d)I�@h�=@̉<X�=�A=`|�;�M�;|Ȝ=m�=�9�= ˓�H��<r(��/�T�<�9.��־=���<X�\=`м<��N/(�d���pG�<Mt��  +���o��E�<���<����䚼�?P�P6Ža�$=@�<܇$��Հ<|dt�F?��	^=�0� ��r;>�ֽ4M)����'�;)%���I�<`ֈ��� =�I���\��@ 4<�.c�.I���� <jV�|z�d��<�. ��/����<򨚽 g�;(A�<��=�[���Z��,����b����`ym=Hyۼ�t��r�>ks=�_{�����6���K���ݼx�ؼ�38� �r:�|]=@�5<����+��S��9��;�=�� �Pѽr%�����y:��w�I�>H�<�q��Dw�;%>e�~
P��V�ʼ1f�=w�`<��;��;<�G� ���틽n�8�u �l$	��be=���= K޽*-=(Z�<sp��aW�=�1l���7��w�丹=(��<���;�s潸I8�辏�������;��<�6�=psp�\��<��'����<�B=�ڷ�����n�<|�ļP����1��4��o����<xln��5�<�Ur����<\T�������>,��L�=�:�<h�"<@�?;�m���:/;X�K<��B<��=<ټ�1n=����hj=b��=�=���<��*�`��=p&�X��MC=TD��Φ=���=�<�<�0Ի U� `<���<�����ۼ m⼆E{�����
��=���=mץ�.?o���ͼĒ��huf<�~
=߲A=[�=@=�n���Ə��#k<���=X�8=�E�<O�e=�Y=���;�N꽨)l<�=�nȼ�� �&s;=	�=��<(�����U=�՝=`؀<�գ=���W�������μ��1=X���T&�)�=��{2�=�<�=%=�4B=��ּ�r<�=�����g�JZW=8�U<�M��N'������AQ=k�=p =����ji-�4㕽*Z�=Ѐ�<���=�-i=@D=�`�"�h��=�t =�#�=�ˤ;P��;����=��/I=0��;>ͭ=��m=��;D<+�?ր��@�� H��Sbڽ B��n�s����$�<fh�h���lf�6v��Pe�<�֫:ЗF��������b=�t��<�D��:�<�X�=3����ݼп$������R��$��*c�N�	=a祽�v��p��<d^��PA��`,/;�L�dX̼ ����
E� (ؼ�N��'���ؒ;~�b=��=�:t�̒��lh��Vf��<�rZ�=(|6����/�>\F�<�e�6��L�������lN޼`�����}�&��Sy�=l��B��p�ϼ��ͽ &d:�f=����E����½l9Y�n3�_ǽg�=ܾ=����l�,^�=��L�`���;�8!�K{�=����L)�<�,����
<p�%��� ��pL�C{���ڼ���0[<�yh=h��F=����yY���1�=�Jy<A�ٽ\
�@��7�g=��Z�K �S,޽�۱�U��mך������=��=P�E����<䣎��aM;h��<PͽM���й�;d���D�'���*��T���u��~;@��:��;`�j����<�ּC�ڽ6��=��';�=�l%= 0P8�sr<L4��`��@�)�H�<2�8=�U��N�@=�\�d�<SEo=���;x�}<�B�ص=Y��Z�1�h��<P��;�A<�h]=~�<�W|�@������<��������Ǽq���Sʈ�k9�=�Y=5J����Y� ����Tz��X�< �<�Q=-w=�z�<C���>o���<�rj==��?=Xq�<@j�<�ku�ս����ʒ0=�q�T6����<U��=��P<Z7����(=���<8Y<x��=J"�6�V��R�����<>�Р.<��T=�Ỽ�b-=-.�=F�<Է�<��꼀��<< �<N������lX5=@DJ;Ц��p�V��q�e=���=��=(O��0������4�Z= H�<v �=)�=k��P��1:~=��=3�= ��: _~�ص��&e��v=�h�<C0=2��=�6��`D=�,ֽ�3(�~g��V�r���� <hH��3F����< 嶻����t��[H��H<�8��TzE���L�4�<4��� ��SGƽ`9�<�n���ٽ@d���~��d���6��eڼ����<����м7>=n����̽����F���������[����!��0�����<Й�;�_=@}<R��`U
��(�� ���,� �&=�]���<;jS�=��<@�¼�Sѽ��$;[ý��g��k���[N������=L���pV�p��������;�9�;0D�te��>���"s���]��mƽ�Ղ=��=B���,�;�J�=��A�Zi�P�;�Ӄ<.��=`L����<�鍼쫐<�`y��O�� }�:w�˽�ꃼа������`�<��Ľ4�<�V��>>���a=\�.=?"� ��01C��<�Y�
�f�ӛ�� ��,&���A����ݼ�A�=$t���¼x��<`��̨��h'u��$��I��� ��9*m�p8p�$6x�XE(��ֶ�`p����A<��� ؔ�[z8=ԡݼ�
Ϧ= �d�'hu=��X=��?<���Nb�.D��S'���<��[=ZGI��[=�j�П�<�nn=<���P���`>)�$��<N*��KU��< ,���]��sE =�\!=��R� H��@��Ȭ<{�dD�t�*�|���^�����=\�?=[Cǽ������<D�A�xP<����
m<8\=PT���#�f�\���M;���=�[=l!$=���<�A�P�>�䙼`�H<P�1��C��@`:�~�=��*��ė���%=�}�<�X<:L=n�%��췽8S_�F	x�d��<ޤ��iI<<
�<�Y%��<��=3�<�{�;X��R�=@��ǽ����cR=�F�LB��h�Q���e��H����=�G= �O��
%�0ˀ�+rA=@�n;���=s!o=>����-d&=$5�<@��= �; ��.@��	j��4d<<YJ�H�J�pgW��M>�ص���q=�=�=<ݼ�S=#P�=hwS���$� ��ڡ�?y=�7=�,=�-����V����<p������C���T�в�;��l=�A��,��=]98=�ս�Aw=2��Cq����=��<�=���Qٽ��`4v<� ;Ȑ<�Qu<*Ml��8�=�i�=��꼰b�<�b� ���"p ���=t��=��ܼP�Z<�nN��=�=������];PC�<n���o�=S�?��R/��T��G�=�qj�he=�\�=��;�K=LU]=P<����im�(o��k���'-���=��`m=V=�I�=���Ą)�\�<���=D<W���Z��S˼ KR�ƫ}�5ó������/�<P�Z���0��+�;�3�#�˽x[<t=9�0=9P�=�n�<)��=6Yd�8�y�:���Dr�<駼(�C�(�<��<���=r3��u�=�4�(Ol=�w=(>>�؂�= �:�V�����:� )<P� =�(�����R�1=pތ�@`<@�: �A9-4�`:5���� <������O��;\����r=�:��|��<�"���<0�ؼ�廈�s� ��9�_�;����E��`U�<@9-�P�< ��<4C�<P<`8V;]�&=�Ç<|�ż��#=��`|;
Y)= *�;��<�� :�t�@'o; 4�������R�|�������֯=]�d=^�B�6�/���; څ�N����< t��Զ�<h�#<����\���c�Yd�=,�<p�W���=��=pd�<�G��أټ��z;@�R<@E7=p�b<��S=xP��@>^����<�l�=��Q�d��=J���8����<:���|=�;��p�i<�P1�`g�<�O=�%3�\��<��<@%E<�k�;@/
���W�ȿ}=d��< �ٺ@�f;p�ܼ ��; M<�;V=葉���N��1����d=��	�})=9�=��<Pԇ<��=�N�<d1�=��I<\H�<��ѽ��[����< �B�]�=y�4� >v���������F=̹̼��^=��b=0�b�*�?��zL=TM���<0R<�b+<|,ʽH�|<�5#=X�3�8S���~�u��ɪT=+'=�����E/>hvW��^۽�\=�]���#��=�������c��.��B�p� :;|��� z�;(�f<��"�5�=���=��<�G =����l<�=n>�=8�a����<d�r�RK
= ��º��<��,; �=)�8R[�ǈ��[��d�<�r����<�=tԼ��o=D�<�r�\��z�b�H��<�9�Ew���J�=@Aм���=Շ;=�">i���p�@�P�ݼ��>�Ʊ���,�r�&�ήI�u�'=2X����P�Ż0��<��Ƽܷ��tBQ�l��)����N��m�=��=Sj=��=��=98��$��<�}�`)=(����
�~\�=t�= d�=�*���ˋ=Zch��ʒ=�20=��:���>|���M������V<�X$=�8�<H���=4ؗ<��$<��p<�I���� <��<ގ�pW�<��9����Dp=�"���=V�\� �7<XG[��z5<ܹ�� v:�뀼`�n���s;p����	<���;@�8<p/<80�<�<��=���<��W=p�l< ��c3=�w��(�8<��7=��O� 
�< Qp:�a��0)�<�l�;���<�Ϡ<����<��b�d=�o=�x]��OD�P����έ����&2/=HIq<@��:��R������Ҽ &��yԊ=�
�:@�o<�U�<��F=���<�������F=�<�=��<ȧ�<`�� ����(w:x��= �W9��=�g��n�T��<�a޼��;=�27��Տ����<PE�����<��\=��j<?A!=L��<��C< �;�ݕ<@N�o�d=���<���<@t�0�h�8�<��<T�;=@$��D ̼`)��lQ�=�Au<8v�<@��;��'<P�λa�M=~eP=�f0= ���WE7=�a��\"C�X��<��F����=,-%�uz>�����!�81p�
-�?��=�<�?i�[T��N�=���p����ּ.6���2�y=|=D���TT�j%ڽ�	=�E��=0��;X�(�yz>�����ȽJu=1�����)�=�1����A=��ٽ)�	���\��������;P��;����|�=.��=�"�<���<�<8��<�q%���/>�\=�3b�x<�`���Ƚ��L=h<�@-�P�=^�=j�:Y��2Ͻ��5�����G� ����J�=01�7�=p=��\M��+ռ��0�;��=04�Pν(!�<,�
�Ģ8=@3�:��U>z��Xm]�f���B�9>��\�0�����Q͋��=. �.�1�`�;L�<��1� �
:<_�Z��ܽ����=���=�A��X�O=/S�=$���ɢ= _��P=��@�X��<䩿=��=��=���D�<���� I= �<@���J>L�Z����X»8��<kl_=`�������!�< .�;��� ��8PWU� ������<,���Dm�<�wm�r�<hq<<-˼���=�V�c&=�-<��<�0��V �p,� ����2;\4�<PK�;��<�K�<�!�<��K=i�=)=�I:>u�=(yd<�7��a= ���~�= �y=�<8a<�B����<,�!=x\�H]�<Dr�<,(����=Z�==
���4��4�����B��x�@,E=W=���<@?";�ˇ�Z'0��x�<�h�=�<�4=�P=l�r=H��<J�E���λG�%=���; ��<�X=��= =���f'��[<���=��<���=� ݺ�@�:<�~��:.d=t@���<�M=0���>=@ې=ؔ�<:c=��J<���<l޾<8���Z��|m=6@=X�<h���4�$�@u�<{�l=Q�,=�lH�Tc��􂄼U*�=D��<��u=|<�<ظ!<x��E�=�=[�J= k;��;=<Rf����O	=T�"��>�7���F�=�#��b�e���{���)��l~=������K����~>�.��w��81�Up��?�$�=�O�<�R��5���ǽ>Eg��Q�=lQȼ�J.< 2v>r'�����ṕ�L༴�Kf=~桽.�r=_G߽����e$�Ԍμ"*�����;8z ������e=�y=Ps�;�����*�@�K;z�;0�> ����������n���E�μ�=@s�����@M<>dK�</-��|�p�/F����H�I������d�Q��79=�@��x�/�.�X�)�~
:���鼓�>�^3�C!Ƚ"��h�)�`9�;PD���=>��;��J�ֶ��CQ/>�8a�Ł��H�A�G��r&>|��0活 �;�)˺��l�@�G<�C��v�3��'����G�b4(=�)>8h/�-�P=��=��%�;P�= ����*�;b�y��>D<k�=&��=�h=
!�8b���bd�p��;��<���<3<>�_�+�< 7ջ]�!=MZ=�t�k��t�<@�������|���߼�<���<0V��pE�;І���=��g�S�����=��T��c�=Ϟ=HZm<�FԻ�$� w��mŻ`p�L�=��֩F= �+<�S=�5�=�%+=k9=�9�rڍ=  �n��pR=<ڣ�h�6=��=�ߝ<`Nr�t'��y<<V"=V�+�@��;pG����!�U䆽t��=i��=cٻ�R0��Y�#����� =݁&=�Mx=�c�; �?5���
�<cȯ=��3=^�=Wq=�U=P{�<������<�J=�T��`��;V^4=�t�=x�,<Z`���g�<D��=`2F<[�=(W���ἰ��@s����`=�ۍ����;bҘ=P/��	��=d�=��
=)U=��(��R�<@.�< b��n����=�^�<�Ӽ����t���&�<x��=��2=<� �V��7>��q=x�
=/"�=�*]=��;��&�n�=h�p=��=p��;��=��e�\1��::=8�b���>�9=0�U=��?��oX�����Nu+��=�g}�(�k��+�����=p����Rf����#���������=��<�lFw�����X�C�,�M���=&�e�|�<�*)>:B��ߌ���5�D����v���D�={.=�!���rݽ��:<�|&��ɽ��������.����<P��;��%��5��̱Ͻ$�����<�޽=�E?����,͓�(O<��}W����=�5>��)����=>�W����C�,�{����tE1�/	���]q������ef�=Tr���@� �޼퇧������=4��%�������tbG��j��'G���7�=�x@=�Y:�V#��zQ�=F�:�|�����8]���> <���<��;�����Pm� t�;tR���cY�6�!��5A�0А����=��2�gvG=-
��n�����=�l<���^��Ц����==�D�f���FR�v
�����sI<�'�=��=8!�|��<8�m�ؔ�<���<X��z�z��w�;0S��*����.�@o�� zX< ��; �ٷ���=�3��m���c�=����,'z=��U=P��;�_Z; (]� �	����0�s<�d!=!���A=��*<���<Ö=��T<<ؒ<h����r=�sf����li�<��;t&�<�k=���<]� ���4:<��<&yL� �ܺh3*�>�n���]��$|=r=	�ƽ����ͻ��e���'<��u<�=46= ~ѻBqj�z�;�4�<W|=9�7=��v=�~�<��<�Ռ;u���@R�:(�C=hw���Қ�䙗<��=�c�;X�x��*�<r2=���<l�=���C�`�J�b�"���:=@�f�Qo=%.q=(�~�ę=�@�=���<�y�< ��� �<���<&g�����~�z=�A�<��R��xO��:g��ё<4��=O,.=�7ʼd�ټ pG�N�+=H��<�ÿ=��=����@U^�=�= �5=�ԡ=�@�;l��<�i=���L�H�Z=P1<��=DÆ=���;����{(���l�h>�p�\<]�ʽ 1�:pN�J=8�:<d�Ƽԟ7�qҀ���N��y=�����`�������;����R�I>���v7=S�=7�
����d戽�P���������(�ۼt� =����6��� .�<r����1Žp���V�8��Лa�8S	���伿aŽ��+�� *�c�<��5=T�~�(z-��ؽ�8	�fA?��Q�=4�e�\O/���>�x%��B̼`����p�� x��>����et�:���2�=nq�$�B� _���@�;�r=�^��:4������B���P�����V�=y�a=h�@���o�'�k=�V��]���)�L����=�P�<(�< vF�@�(<����YE�Ģ��,�������� �Hc+�|��=���>F!=Fo>��2����=L��<�v���������*=(1�����uƽ�m��|����� ��`γ=dy�<��T�'�<|�`��; g��#Ž�В��ݻ������&j��Љݻ�������X}�<�5��m]�Ad= ����G��J�= 5��7o=b�=�:@<��6�Pym���8��c�H�<M==��+�WZk=�����'�<�m�=�g�@5���ټ���<HKڼ"��84\< �)�h�$���5=H�;=�3���햼�PI;��<.2`�lEļ4�7j��"Vy���=�h=��vm���<��6��D<PE��୭<�g%=HJǼAE����0��?
<Tm�=��U=na=p�.<��컰� �=��0p&���<��-���� 0���ķ=�#�(���= ��<��<��U=�XּI9��@Ո������=@-����*=��=����i�<�P
>�/�<�t�;��ܼ��=�l&<L�ν���rs�= έ:�߳��+e�Hsl�h�d�L�>� O=�㼒m+�(�Y��=��A<�@�=�j=���� d���=x�<To�=P��;�?D�:���8w��p��<X�����o<|���>����=b�F=�
� [�<B��=���#���=Jn� ��9`��;2`J=�3���r���#=��/;��L����
^N�^�A=h!<=��8��=�,�<�H����=^*�;Y���J=p=�<����OO�,x���������*�;�\�<L��<6<����=���= <��Ǽ�$����<�l���=���=�������:��^�1O�=���K����(=r���\S=��t������۽�H7=�Ǝ��n=B�=�j��Af=�ٵ<J=$=���������#�X���f��X��=<��D��<���<�l�=����:�p>�;{�g=��;x㓼�,��Mм�#�5|�������S���=ܚ��`;�8Es��b��8 �� ����=R�=⩁=�~�D�h=�~��`TL;��L�,�=T�ɼ�i� (h��h�=�j�=*IK����=�ס��|�=K�c=2ҋ���=X�޼`b�O=�x<���<�7��X��tb=`_�5�<�@<\S�<j��𻎻p�Ҽ0x���2R<�j�<(�q<�t�;K���}�;�P<<�=�ռ:�@�޻@�o� #`�@��:�f��X��<�傻��<`	�< �l<�Z�;)C�0��<��<@Sٺ�w�<���8<�8�<p/һ.A= AW<��9�H|b<\*=��; _ܻK<�F����=��=.8I�FY�`�;@T%������<����Tt�<@Tͺ;�����S���f�=|��<H�T�p=X=g<�N�<Ri2�����������$W=D�<J�=H�J���ļ@6d;iE�=X�c���=h[Q��Ѽ
M�=�����<�Ϡ<��ϼd:�<�c���=�B�<@��:�ׁ<`��; /<�e���o;�-�;�hv=�D,=�<�O�<(m�HKݼ��ۼ��{=����Jż�5<��=�Ä��q=	5=��< k����<���<�= ��<	!=,���вy�D��<�����=�B��c02>td=�l�E�XlD<��*���7=<x�=��/�8���^�=WӍ�fA�����|��<��н�\��G=``=;^������ܽR��=�`;=�*��V$> �׽�;=J�E�#��0σ=$ ��@.�<�����b���͠�X�W�(�p<�=�sE��/�=��>��=�׏��� ��E�<�=�*�/>fN�=��k�t#�<���l<�!1��*� �<ؤz<�@=m��T�$��d���0#����6"=�+>Z_Խ]�r=�m�@��<]��������<X���g�0�=����e=�[�<�?9>�@ҽ���-���=������D�����y��i6<B�v�Q1��t���䮼<0���K<�ߤ�Vc��K���ɼ%�=xC�= ��:@�;&~�=����	=�,J�ˊ=�Խp:�<��=n��=x,�=U���8�=s�սL;�=ܦ6=fݖ���0>�;T��hż��<Ж�<���<���< 	H<�b�<���<��z<�P�<Lj�<��< ��<&��0�q<@p<�U[<a==��<�ۿ;
�t������=/<l��< ͩ���@�O�Xs����$��(,<8y<��;ء�<�)<���<�"�<��<���;2L)=��<�+�<(
=��w�0"7<��<�e�<,=�J�<t��<��<�&+=���<�J�<8�< ��:��C=584=.�I�L�����q� x�8��s�= ą��b^��%�xt�<ؠ� iN;��=@�s;�jA<��C=��<D/�<�4�@�G�0��@i<�s+=���<�u<<�䡼X^�������=(�'��%`=8J�`0<l��=�0O�E%6=`�< �;:l��< ��7�
=,p=���<.I=(p�<���<0��0%�<@<v<�g=W5=m�9=P�<`��;h2���C���m= �8;�mT��G�<�FC=���<$��<�7�<`9�<Pq|����<gzS=�`�<|`�<�ER=��x�ȨE��c�<����L�>(̽��3>p�l���O�Z
B�^�6A�=��<=N��q\���,,>�'j��^ͽ@�,�aX��齁�Q=(=缨����A��M��		>�`�<��O�]��>`��o�� ��<�XY�t�.�dr=ܴ����=������~�=�4Kļ�[T�0٣��9=��<_&�=�>��5=تq��g��ex<�'�$�C>���<���P�W<�Ͷ�y����=�x���Sc���>�N<������5��\&�x\�_䏽z�n� ��� f�=�\��3�<`�f�t�������b~�$�
>5\R�QϜ���<��(O=��@�2�h>bZN����㽦6>�y������>��AϽ8��=�����Z<x5<$?Y���<��֨R�����lb�	�=9.>����~�<�W�=V�@���=I�����=F�Z�t�=�Y^=#�'>�f�=
e�`��<�7��7�=0r�<<�S�D8�>���P|	�� <�=>}%=E�;����T)�<0��;P�*���<,��<�i1����<���Ȇ�<O�; ��<�=@V#��N@=Z���<���<(��<lX��@�ûA#��&���,�$��<��;L��<h��<��<},=Z=$�=�����=�ȶ<@^;T�K=,���h��<;�g=��<���<P٤��U�<0}.=L��<�#=�v�< u�:X�n��h=$��=�z�J�\Hͼ$��@x=�65B=@Q<��<`� @;��ۼ���<�"�=b�<0��</�p=�v+=�=�<j(�@�:��;@�];��<��=s�=�����`r�F.�= T��=`�����<�M=0�R��z=`ge;8X<tMS=�ש��|T=ډ=��%=��;=X'�<�*�<@�R<��u��ʦ;Y�=�5=4k�<��C<��xNI���=_5|= ���5?�@Q<�~l=/�<	rh=^C3=���<�����<g=�m=�)/=�<�fK=PI#�|M�hg�<t�f���*>�����p>�꒽����휽c�O�k=�s�<�:��������@>Q��BܽL)��Ƚ�ӽ3W�= TS<�_[����k���q�%Q�=�3&<�o<���>����蜽hx����/�` 6�Tl�<(���@�=�A���h��\���g�������j�<$�=@�=���=��<�y��T��Ɍ< ks;�Q>�������E�����uB���>|8�L ڽ!G>$ｼ#7��0༴G�X\�'Ž�5�L�T�B��=_e��t,��]��� ��/�����5>k�A�g\������Ȕ�8��<JGE�7>��*<��T�����">卆�я��pǸ�ycؽ� > �; |n;��=(�%�02��Ȭ�<�#��`m��$��B�����}=D<>��_��]=<��<�AD�Z�>H����]=�����9=�u=]f>�=:� �@W�E���==�e�< �P��Pt>����e8�,ρ�G�&=s==<<��l����s<��������¼���;T���h��<x�ټh�<<`f��]�= �b<�[-�\y�=��{�٫7=�h=Xĺ<`�� ���k!� � �߹8��<@�8���==�u�<�O�< �_=f�=̰�<��3�:��=�T<����&pE=�v��u�<��=�,m<@��<��żP��<#�"=��m�h~�<P��;����|Aܼ�=f�=�=����x��|<s�`�$���E=$��<8�=�[[�g��Ξ?�А�<��=�p�<�W=�G=�36=���<6xs���>;`ʛ<����`�<�)�<�R= 4>�d�V���<<���= ���Ϟ=(W_� v#��|<TF��=zl=��T�(��<��l=�$x��]=�w�=˚'=� =`V�;�^�<��<�o��"�=\s=�x�:�3�X�� {;V��=�Mt=L���l���P�V�Wq=d��<v�=0�O=�+<�H�����=��G=6�=P<<�=�3�@l0�T4�<�>���W>���ڶ�=�7��lmx�������=���= �;��"�*1V�Ј>�Զ�����\B�/-�ރ���ż=������m���(� jĽ<{X��%N=��=�l{�<�{D>��t�e��|s����@��������ς=�᜽`���?�;�R�:P����%� ��pm�<�u�<�k=��D;�׀�ս���$<� A<��=|I�4���	��P�W���?����=��x�*����=>.�F�]��W�ߗ��|0��ٖ��y�@^���f<�=Wӽz����;��ف�0�»��>
���f:]�i���,s���+�����Զ=�}l=pf�;�
ؽ�2�=2.�N!1��+��Y���V�>��<l7�<�=@���6ꅽ<a�<
��~iX�D�`qR� ��c�!>f$a��HB=�9�����=$�������Ud����O�R=���=�&�����@ؼ�f��9���ݝ<s$C=�>p�/�P���ȅ�F�<P��<ଲ��߼�; ��:�N��@Q�@Y�<�!k<`g�<��%<@�Z<�"�;Gr#=`��<���V�9=x���<�.+=x�<�\X�F)=8�?���>����<�< :\<lo/=�(�<@_�̅l=��<xDh<�J�<~)=P	<�P�;���<ܻ�<p�;kz=���<��; ���D�<���<�����<䫻<�开_N�L��<l'�=����Hi��=,��%μ ����=RP= �������׼�ռ�9�<�ex=���<B�f=Z�;�e�<�H�<�V!���Ҽ�u<=pM�; ��;@(R�q-G= T>��Jʻ�^<"�&=�/<�s�=����/ּL9�6�h�a�7= _k:}~�=|��<@߹�XAV<�L�=��=���<��A<ha�<��
=`�)�,|���6�=�OJ<�XB������xV�8k
<�H�=vQ]=�dֻ茵��dz��>=��<0V=��<H4�H`I�n M= �
=��=��ѻ%=y%�@�U����<-�;��=��c<���<xz*���M�ov��9L���<�>U���c:Q�t��=@��V�l��CI�3Q۽��N�pH�=X<-�X�H��d��|6� i%�8�J<���	)=`s�=D���8��j蝽�����Ǻ� �G�yO5=z{�D����J�;L-��`���~�S�$]ڼ@��:�2[;�\g�P켼)S�� �����@;�K<�~@=Bm���e?����� _7�b%�3��=���eǽ�z>Y�Τ�� %�C7���B�H�����7��{e�L��={ӽdᒽШ
��멽��6<H��=����4��-�ѽ(�b��K�S㺽�=G��=  8fʢ��)\=� ��f���%ͼ�̔��� > �	=���<d��<�G ��,���;�����L���H�'�L���(�=6/��� =�x�S*�����=��������L �ֱ�Z	= 7;�U��K���>P�K���"���<H�=�X=D���0�ü���� �;@�ɻ���Z���q�p���h��8�g�\3�< ��9Xz<FJ=x
� ]��fK=���;-�� V�<'4<$Z�<��R= �/;��i��a-=�욼䓽��=��R<@���%+=8<x)g�EPP=�.I� r`���<`R< :o:`�;`��;T��<x���D1=�&= �����;�s<�Jv<<ԼPz�;ܼ����>���K�<��z=��|���%e<		� >�9dҳ<�<0����`���Q��[���Qi;�Z=T�<��\=�����
����;:i�ں���'= ��H�� ���,v=H�c���
���<HsP<P�<�
9=:׼^T��{��C�����<���;�|�=pP�; su�h�
��q�=P`	=��û�8�:�G=,v�<��z�D�׼C%�=�0���J"��:��.� P9��>'*W=��<�t�꼔䨼,Z=ؾ/<�Q=̚�<\ʼ`W�"�=ȿ#<C�=�����(�<*
dtype0*'
_output_shapes
:�
�
siamese_4/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:{{`*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_4/scala1/AddAddsiamese_4/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_4/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_4/scala1/moments/meanMeansiamese_4/scala1/Add/siamese_4/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese_4/scala1/moments/StopGradientStopGradientsiamese_4/scala1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese_4/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala1/Add%siamese_4/scala1/moments/StopGradient*&
_output_shapes
:{{`*
T0
�
3siamese_4/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala1/moments/varianceMean*siamese_4/scala1/moments/SquaredDifference3siamese_4/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_4/scala1/moments/SqueezeSqueezesiamese_4/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
�
"siamese_4/scala1/moments/Squeeze_1Squeeze!siamese_4/scala1/moments/variance*
T0*
_output_shapes
:`*
squeeze_dims
 
�
&siamese_4/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_4/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_4/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
ksiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_4/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
 siamese_4/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
(siamese_4/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_4/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_4/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_4/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_4/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
c
siamese_4/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_4/scala1/cond/switch_tIdentitysiamese_4/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_4/scala1/cond/switch_fIdentitysiamese_4/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_4/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_4/scala1/cond/Switch_1Switch siamese_4/scala1/moments/Squeezesiamese_4/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese_4/scala1/moments/Squeeze
�
siamese_4/scala1/cond/Switch_2Switch"siamese_4/scala1/moments/Squeeze_1siamese_4/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_4/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_4/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_4/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_4/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_4/scala1/cond/MergeMergesiamese_4/scala1/cond/Switch_3 siamese_4/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_4/scala1/cond/Merge_1Mergesiamese_4/scala1/cond/Switch_4 siamese_4/scala1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
e
 siamese_4/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_4/scala1/batchnorm/addAddsiamese_4/scala1/cond/Merge_1 siamese_4/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_4/scala1/batchnorm/RsqrtRsqrtsiamese_4/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_4/scala1/batchnorm/mulMul siamese_4/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese_4/scala1/batchnorm/mul_1Mulsiamese_4/scala1/Addsiamese_4/scala1/batchnorm/mul*
T0*&
_output_shapes
:{{`
�
 siamese_4/scala1/batchnorm/mul_2Mulsiamese_4/scala1/cond/Mergesiamese_4/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_4/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_4/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese_4/scala1/batchnorm/add_1Add siamese_4/scala1/batchnorm/mul_1siamese_4/scala1/batchnorm/sub*
T0*&
_output_shapes
:{{`
p
siamese_4/scala1/ReluRelu siamese_4/scala1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese_4/scala1/poll/MaxPoolMaxPoolsiamese_4/scala1/Relu*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
X
siamese_4/scala2/ConstConst*
_output_shapes
: *
value	B :*
dtype0
b
 siamese_4/scala2/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_4/scala2/splitSplit siamese_4/scala2/split/split_dimsiamese_4/scala1/poll/MaxPool*8
_output_shapes&
$:==0:==0*
	num_split*
T0
Z
siamese_4/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_4/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala2/split_1Split"siamese_4/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_4/scala2/Conv2DConv2Dsiamese_4/scala2/splitsiamese_4/scala2/split_1*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_4/scala2/Conv2D_1Conv2Dsiamese_4/scala2/split:1siamese_4/scala2/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0
^
siamese_4/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala2/concatConcatV2siamese_4/scala2/Conv2Dsiamese_4/scala2/Conv2D_1siamese_4/scala2/concat/axis*'
_output_shapes
:99�*

Tidx0*
T0*
N
�
siamese_4/scala2/AddAddsiamese_4/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese_4/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_4/scala2/moments/meanMeansiamese_4/scala2/Add/siamese_4/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_4/scala2/moments/StopGradientStopGradientsiamese_4/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_4/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala2/Add%siamese_4/scala2/moments/StopGradient*'
_output_shapes
:99�*
T0
�
3siamese_4/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala2/moments/varianceMean*siamese_4/scala2/moments/SquaredDifference3siamese_4/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_4/scala2/moments/SqueezeSqueezesiamese_4/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_4/scala2/moments/Squeeze_1Squeeze!siamese_4/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_4/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_4/scala2/moments/Squeeze*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_4/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Nsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_4/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese_4/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
(siamese_4/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_4/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_4/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Tsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_4/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_4/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_4/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_4/scala2/cond/switch_tIdentitysiamese_4/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_4/scala2/cond/switch_fIdentitysiamese_4/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_4/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_4/scala2/cond/Switch_1Switch siamese_4/scala2/moments/Squeezesiamese_4/scala2/cond/pred_id*3
_class)
'%loc:@siamese_4/scala2/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_4/scala2/cond/Switch_2Switch"siamese_4/scala2/moments/Squeeze_1siamese_4/scala2/cond/pred_id*5
_class+
)'loc:@siamese_4/scala2/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese_4/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_4/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_4/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_4/scala2/cond/pred_id*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_4/scala2/cond/MergeMergesiamese_4/scala2/cond/Switch_3 siamese_4/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_4/scala2/cond/Merge_1Mergesiamese_4/scala2/cond/Switch_4 siamese_4/scala2/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_4/scala2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_4/scala2/batchnorm/addAddsiamese_4/scala2/cond/Merge_1 siamese_4/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_4/scala2/batchnorm/RsqrtRsqrtsiamese_4/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_4/scala2/batchnorm/mulMul siamese_4/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_4/scala2/batchnorm/mul_1Mulsiamese_4/scala2/Addsiamese_4/scala2/batchnorm/mul*
T0*'
_output_shapes
:99�
�
 siamese_4/scala2/batchnorm/mul_2Mulsiamese_4/scala2/cond/Mergesiamese_4/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_4/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_4/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_4/scala2/batchnorm/add_1Add siamese_4/scala2/batchnorm/mul_1siamese_4/scala2/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese_4/scala2/ReluRelu siamese_4/scala2/batchnorm/add_1*'
_output_shapes
:99�*
T0
�
siamese_4/scala2/poll/MaxPoolMaxPoolsiamese_4/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
siamese_4/scala3/Conv2DConv2Dsiamese_4/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_4/scala3/AddAddsiamese_4/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_4/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_4/scala3/moments/meanMeansiamese_4/scala3/Add/siamese_4/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_4/scala3/moments/StopGradientStopGradientsiamese_4/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_4/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala3/Add%siamese_4/scala3/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_4/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala3/moments/varianceMean*siamese_4/scala3/moments/SquaredDifference3siamese_4/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_4/scala3/moments/SqueezeSqueezesiamese_4/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_4/scala3/moments/Squeeze_1Squeeze!siamese_4/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_4/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_4/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_4/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
ksiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_4/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_4/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
(siamese_4/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_4/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_4/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_4/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Ksiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
"siamese_4/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
c
siamese_4/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_4/scala3/cond/switch_tIdentitysiamese_4/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala3/cond/switch_fIdentitysiamese_4/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_4/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_4/scala3/cond/Switch_1Switch siamese_4/scala3/moments/Squeezesiamese_4/scala3/cond/pred_id*3
_class)
'%loc:@siamese_4/scala3/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_4/scala3/cond/Switch_2Switch"siamese_4/scala3/moments/Squeeze_1siamese_4/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_4/scala3/moments/Squeeze_1
�
siamese_4/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_4/scala3/cond/pred_id*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_4/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_4/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_4/scala3/cond/MergeMergesiamese_4/scala3/cond/Switch_3 siamese_4/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_4/scala3/cond/Merge_1Mergesiamese_4/scala3/cond/Switch_4 siamese_4/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_4/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_4/scala3/batchnorm/addAddsiamese_4/scala3/cond/Merge_1 siamese_4/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_4/scala3/batchnorm/RsqrtRsqrtsiamese_4/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_4/scala3/batchnorm/mulMul siamese_4/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_4/scala3/batchnorm/mul_1Mulsiamese_4/scala3/Addsiamese_4/scala3/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_4/scala3/batchnorm/mul_2Mulsiamese_4/scala3/cond/Mergesiamese_4/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_4/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_4/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_4/scala3/batchnorm/add_1Add siamese_4/scala3/batchnorm/mul_1siamese_4/scala3/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_4/scala3/ReluRelu siamese_4/scala3/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_4/scala4/ConstConst*
_output_shapes
: *
value	B :*
dtype0
b
 siamese_4/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala4/splitSplit siamese_4/scala4/split/split_dimsiamese_4/scala3/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_4/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_4/scala4/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_4/scala4/split_1Split"siamese_4/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_4/scala4/Conv2DConv2Dsiamese_4/scala4/splitsiamese_4/scala4/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_4/scala4/Conv2D_1Conv2Dsiamese_4/scala4/split:1siamese_4/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese_4/scala4/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_4/scala4/concatConcatV2siamese_4/scala4/Conv2Dsiamese_4/scala4/Conv2D_1siamese_4/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_4/scala4/AddAddsiamese_4/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_4/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_4/scala4/moments/meanMeansiamese_4/scala4/Add/siamese_4/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_4/scala4/moments/StopGradientStopGradientsiamese_4/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_4/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala4/Add%siamese_4/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_4/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala4/moments/varianceMean*siamese_4/scala4/moments/SquaredDifference3siamese_4/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_4/scala4/moments/SqueezeSqueezesiamese_4/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_4/scala4/moments/Squeeze_1Squeeze!siamese_4/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_4/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_4/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_4/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_4/scala4/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_4/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_4/scala4/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_4/scala4/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_4/scala4/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
usiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_4/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_4/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
c
siamese_4/scala4/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_4/scala4/cond/switch_tIdentitysiamese_4/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala4/cond/switch_fIdentitysiamese_4/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala4/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_4/scala4/cond/Switch_1Switch siamese_4/scala4/moments/Squeezesiamese_4/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_4/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_4/scala4/cond/Switch_2Switch"siamese_4/scala4/moments/Squeeze_1siamese_4/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_4/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_4/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_4/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_4/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_4/scala4/cond/MergeMergesiamese_4/scala4/cond/Switch_3 siamese_4/scala4/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_4/scala4/cond/Merge_1Mergesiamese_4/scala4/cond/Switch_4 siamese_4/scala4/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_4/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_4/scala4/batchnorm/addAddsiamese_4/scala4/cond/Merge_1 siamese_4/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_4/scala4/batchnorm/RsqrtRsqrtsiamese_4/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_4/scala4/batchnorm/mulMul siamese_4/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_4/scala4/batchnorm/mul_1Mulsiamese_4/scala4/Addsiamese_4/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_4/scala4/batchnorm/mul_2Mulsiamese_4/scala4/cond/Mergesiamese_4/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_4/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_4/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_4/scala4/batchnorm/add_1Add siamese_4/scala4/batchnorm/mul_1siamese_4/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_4/scala4/ReluRelu siamese_4/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_4/scala5/ConstConst*
_output_shapes
: *
value	B :*
dtype0
b
 siamese_4/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala5/splitSplit siamese_4/scala5/split/split_dimsiamese_4/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_4/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_4/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala5/split_1Split"siamese_4/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_4/scala5/Conv2DConv2Dsiamese_4/scala5/splitsiamese_4/scala5/split_1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_4/scala5/Conv2D_1Conv2Dsiamese_4/scala5/split:1siamese_4/scala5/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_4/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala5/concatConcatV2siamese_4/scala5/Conv2Dsiamese_4/scala5/Conv2D_1siamese_4/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_4/scala5/AddAddsiamese_4/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
O
score_2/ConstConst*
dtype0*
_output_shapes
: *
value	B :
Y
score_2/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_2/splitSplitscore_2/split/split_dimsiamese_4/scala5/Add*
T0*M
_output_shapes;
9:�:�:�*
	num_split
�
score_2/Conv2DConv2Dscore_2/splitConst_2*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
score_2/Conv2D_1Conv2Dscore_2/split:1Const_2*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC
�
score_2/Conv2D_2Conv2Dscore_2/split:2Const_2*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
U
score_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_2/concatConcatV2score_2/Conv2Dscore_2/Conv2D_1score_2/Conv2D_2score_2/concat/axis*&
_output_shapes
:*

Tidx0*
T0*
N
�
adjust_2/Conv2DConv2Dscore_2/concatadjust/weights/read*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
i
adjust_2/AddAddadjust_2/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:"�G&�