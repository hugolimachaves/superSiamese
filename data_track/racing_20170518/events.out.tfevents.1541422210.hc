       �K"	  � ��Abrain.Event:2�V�
SP     ����	��� ��A"Ơ!
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_1Placeholder*
shape:��*
dtype0*(
_output_shapes
:��
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_3Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
M
is_trainingConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
O
is_training_1Const*
value	B
 Z *
dtype0
*
_output_shapes
: 
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
?siamese/scala1/conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *���<
�
Hsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala1/conv/weights/Initializer/truncated_normal/shape*.
_class$
" loc:@siamese/scala1/conv/weights*
seed2	*
dtype0*&
_output_shapes
:`*

seed*
T0
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
VariableV2*
dtype0*&
_output_shapes
:`*
shared_name *.
_class$
" loc:@siamese/scala1/conv/weights*
	container *
shape:`
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
 siamese/scala1/conv/weights/readIdentitysiamese/scala1/conv/weights*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *o:
�
=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala1/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: 
�
6siamese/scala1/conv/weights/Regularizer/l2_regularizerMul<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala1/conv/weights
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
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala1/conv/biases*
	container *
shape:`*
dtype0*
_output_shapes
:`
�
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`
�
siamese/scala1/conv/biases/readIdentitysiamese/scala1/conv/biases*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
_output_shapes
:`
�
siamese/scala1/Conv2DConv2DPlaceholder_2 siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:;;`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese/scala1/AddAddsiamese/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:;;`
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
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *)
_class
loc:@siamese/scala1/bn/beta*
	container *
shape:`
�
siamese/scala1/bn/beta/AssignAssignsiamese/scala1/bn/beta(siamese/scala1/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`
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
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma
�
siamese/scala1/bn/gamma/AssignAssignsiamese/scala1/bn/gamma)siamese/scala1/bn/gamma/Initializer/Const*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma
�
siamese/scala1/bn/gamma/readIdentitysiamese/scala1/bn/gamma*
_output_shapes
:`*
T0**
_class 
loc:@siamese/scala1/bn/gamma
�
/siamese/scala1/bn/moving_mean/Initializer/ConstConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
siamese/scala1/bn/moving_mean
VariableV2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name 
�
$siamese/scala1/bn/moving_mean/AssignAssignsiamese/scala1/bn/moving_mean/siamese/scala1/bn/moving_mean/Initializer/Const*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(
�
"siamese/scala1/bn/moving_mean/readIdentitysiamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`
�
(siamese/scala1/bn/moving_variance/AssignAssign!siamese/scala1/bn/moving_variance3siamese/scala1/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
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
siamese/scala1/moments/meanMeansiamese/scala1/Add-siamese/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
#siamese/scala1/moments/StopGradientStopGradientsiamese/scala1/moments/mean*
T0*&
_output_shapes
:`
�
(siamese/scala1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1/Add#siamese/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
1siamese/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1/moments/varianceMean(siamese/scala1/moments/SquaredDifference1siamese/scala1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
siamese/scala1/moments/SqueezeSqueezesiamese/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
 siamese/scala1/moments/Squeeze_1Squeezesiamese/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
$siamese/scala1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
3siamese/scala1/siamese/scala1/bn/moving_mean/biased
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`
�
:siamese/scala1/siamese/scala1/bn/moving_mean/biased/AssignAssign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container 
�
>siamese/scala1/siamese/scala1/bn/moving_mean/local_step/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepIsiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
<siamese/scala1/siamese/scala1/bn/moving_mean/local_step/readIdentity7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readsiamese/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMul@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub$siamese/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
isiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biased@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Lsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/x@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
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
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zeros*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Msiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala1/siamese/scala1/bn/moving_variance/local_step
VariableV2*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub&siamese/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
ssiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
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
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x&siamese/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese/scala1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
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
siamese/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1/cond/pred_id*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`*
T0
�
siamese/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1/cond/MergeMergesiamese/scala1/cond/Switch_3siamese/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese/scala1/cond/Merge_1Mergesiamese/scala1/cond/Switch_4siamese/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
c
siamese/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala1/batchnorm/addAddsiamese/scala1/cond/Merge_1siamese/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
j
siamese/scala1/batchnorm/RsqrtRsqrtsiamese/scala1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/mulMulsiamese/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/mul_1Mulsiamese/scala1/Addsiamese/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
siamese/scala1/batchnorm/mul_2Mulsiamese/scala1/cond/Mergesiamese/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/subSubsiamese/scala1/bn/beta/readsiamese/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
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
siamese/scala1/poll/MaxPoolMaxPoolsiamese/scala1/Relu*
ksize
*
paddingVALID*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides

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
Hsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala2/conv/weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:0�*

seed*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
seed2w
�
<siamese/scala2/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala2/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
8siamese/scala2/conv/weights/Initializer/truncated_normalAdd<siamese/scala2/conv/weights/Initializer/truncated_normal/mul=siamese/scala2/conv/weights/Initializer/truncated_normal/mean*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�*
T0
�
siamese/scala2/conv/weights
VariableV2*
dtype0*'
_output_shapes
:0�*
shared_name *.
_class$
" loc:@siamese/scala2/conv/weights*
	container *
shape:0�
�
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�*
use_locking(
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�*
T0
�
<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *o:
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
VariableV2*-
_class#
!loc:@siamese/scala2/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
!siamese/scala2/conv/biases/AssignAssignsiamese/scala2/conv/biases,siamese/scala2/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala2/conv/biases/readIdentitysiamese/scala2/conv/biases*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
_output_shapes	
:�
V
siamese/scala2/ConstConst*
dtype0*
_output_shapes
: *
value	B :
`
siamese/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/splitSplitsiamese/scala2/split/split_dimsiamese/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
X
siamese/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/split_1Split siamese/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
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
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
\
siamese/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala2/AddAddsiamese/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
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
VariableV2*
shared_name *)
_class
loc:@siamese/scala2/bn/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/beta/AssignAssignsiamese/scala2/bn/beta(siamese/scala2/bn/beta/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta
�
siamese/scala2/bn/beta/readIdentitysiamese/scala2/bn/beta*
_output_shapes	
:�*
T0*)
_class
loc:@siamese/scala2/bn/beta
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
VariableV2*
shared_name **
_class 
loc:@siamese/scala2/bn/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma*
_output_shapes	
:�*
T0**
_class 
loc:@siamese/scala2/bn/gamma
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�
�
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
"siamese/scala2/bn/moving_mean/readIdentitysiamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
(siamese/scala2/bn/moving_variance/AssignAssign!siamese/scala2/bn/moving_variance3siamese/scala2/bn/moving_variance/Initializer/Const*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
siamese/scala2/moments/meanMeansiamese/scala2/Add-siamese/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/mean*'
_output_shapes
:�*
T0
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
$siamese/scala2/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
:siamese/scala2/siamese/scala2/bn/moving_mean/biased/AssignAssign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape: 
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
<siamese/scala2/siamese/scala2/bn/moving_mean/local_step/readIdentity7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readsiamese/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMul@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub$siamese/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
isiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biased@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Fsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepLsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Asiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/x@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivAsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container 
�
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zeros*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(
�
<siamese/scala2/siamese/scala2/bn/moving_variance/biased/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container 
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read siamese/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Rsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x&siamese/scala2/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivGsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
siamese/scala2/cond/switch_fIdentitysiamese/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala2/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala2/cond/Switch_1Switchsiamese/scala2/moments/Squeezesiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*1
_class'
%#loc:@siamese/scala2/moments/Squeeze
�
siamese/scala2/cond/Switch_2Switch siamese/scala2/moments/Squeeze_1siamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese/scala2/moments/Squeeze_1
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
N*
_output_shapes
	:�: *
T0
�
siamese/scala2/cond/Merge_1Mergesiamese/scala2/cond/Switch_4siamese/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese/scala2/batchnorm/addAddsiamese/scala2/cond/Merge_1siamese/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
k
siamese/scala2/batchnorm/RsqrtRsqrtsiamese/scala2/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese/scala2/batchnorm/add_1Addsiamese/scala2/batchnorm/mul_1siamese/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
m
siamese/scala2/ReluRelusiamese/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese/scala2/poll/MaxPoolMaxPoolsiamese/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
>siamese/scala3/conv/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@siamese/scala3/conv/weights*%
valueB"         �  *
dtype0
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
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<*
dtype0
�
Hsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala3/conv/weights/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
seed2�*
dtype0*(
_output_shapes
:��*

seed
�
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��*
T0
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
siamese/scala3/conv/weights
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��
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
 siamese/scala3/conv/weights/readIdentitysiamese/scala3/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
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
=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala3/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
_output_shapes
: 
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
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala3/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
!siamese/scala3/conv/biases/AssignAssignsiamese/scala3/conv/biases,siamese/scala3/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala3/conv/biases/readIdentitysiamese/scala3/conv/biases*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
_output_shapes	
:�
�
siamese/scala3/Conv2DConv2Dsiamese/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
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
use_cudnn_on_gpu(*
paddingVALID
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala3/bn/beta*
	container 
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
)siamese/scala3/bn/gamma/Initializer/ConstConst*
dtype0*
_output_shapes	
:�**
_class 
loc:@siamese/scala3/bn/gamma*
valueB�*  �?
�
siamese/scala3/bn/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala3/bn/gamma*
	container *
shape:�
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
siamese/scala3/bn/gamma/readIdentitysiamese/scala3/bn/gamma*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
_output_shapes	
:�
�
/siamese/scala3/bn/moving_mean/Initializer/ConstConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
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
$siamese/scala3/bn/moving_mean/AssignAssignsiamese/scala3/bn/moving_mean/siamese/scala3/bn/moving_mean/Initializer/Const*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
"siamese/scala3/bn/moving_mean/readIdentitysiamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container 
�
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
siamese/scala3/moments/meanMeansiamese/scala3/Add-siamese/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
#siamese/scala3/moments/StopGradientStopGradientsiamese/scala3/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala3/moments/SquaredDifferenceSquaredDifferencesiamese/scala3/Add#siamese/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
1siamese/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/varianceMean(siamese/scala3/moments/SquaredDifference1siamese/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
siamese/scala3/moments/SqueezeSqueezesiamese/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�
�
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala3/siamese/scala3/bn/moving_mean/local_step
VariableV2*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape: *
dtype0
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
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readsiamese/scala3/moments/Squeeze*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biased@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x$siamese/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/x@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivAsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zeros*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/AssignAssign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepMsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala3/siamese/scala3/bn/moving_variance/local_step/readIdentity;siamese/scala3/siamese/scala3/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Gsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese/scala3/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala3/cond/Switch_1Switchsiamese/scala3/moments/Squeezesiamese/scala3/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_2Switch siamese/scala3/moments/Squeeze_1siamese/scala3/cond/pred_id*3
_class)
'%loc:@siamese/scala3/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
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
siamese/scala3/batchnorm/RsqrtRsqrtsiamese/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/mulMulsiamese/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/mul_1Mulsiamese/scala3/Addsiamese/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
�
siamese/scala3/batchnorm/mul_2Mulsiamese/scala3/cond/Mergesiamese/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
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
siamese/scala3/ReluRelusiamese/scala3/batchnorm/add_1*
T0*'
_output_shapes
:

�
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
Hsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala4/conv/weights/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
seed2�*
dtype0*(
_output_shapes
:��*

seed
�
<siamese/scala4/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala4/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
8siamese/scala4/conv/weights/Initializer/truncated_normalAdd<siamese/scala4/conv/weights/Initializer/truncated_normal/mul=siamese/scala4/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
siamese/scala4/conv/weights
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala4/conv/weights*
	container *
shape:��
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
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala4/conv/biases*
	container *
shape:�
�
!siamese/scala4/conv/biases/AssignAssignsiamese/scala4/conv/biases,siamese/scala4/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
_output_shapes	
:�
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
siamese/scala4/splitSplitsiamese/scala4/split/split_dimsiamese/scala3/Relu*
T0*:
_output_shapes(
&:

�:

�*
	num_split
X
siamese/scala4/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
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
siamese/scala4/Conv2DConv2Dsiamese/scala4/splitsiamese/scala4/split_1*
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
�
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
\
siamese/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/concatConcatV2siamese/scala4/Conv2Dsiamese/scala4/Conv2D_1siamese/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala4/AddAddsiamese/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
(siamese/scala4/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala4/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
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
siamese/scala4/bn/beta/readIdentitysiamese/scala4/bn/beta*)
_class
loc:@siamese/scala4/bn/beta*
_output_shapes	
:�*
T0
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
siamese/scala4/bn/gamma/readIdentitysiamese/scala4/bn/gamma*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
_output_shapes	
:�
�
/siamese/scala4/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�
�
$siamese/scala4/bn/moving_mean/AssignAssignsiamese/scala4/bn/moving_mean/siamese/scala4/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
"siamese/scala4/bn/moving_mean/readIdentitysiamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
(siamese/scala4/bn/moving_variance/AssignAssign!siamese/scala4/bn/moving_variance3siamese/scala4/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
&siamese/scala4/bn/moving_variance/readIdentity!siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4/moments/meanMeansiamese/scala4/Add-siamese/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
#siamese/scala4/moments/StopGradientStopGradientsiamese/scala4/moments/mean*'
_output_shapes
:�*
T0
�
(siamese/scala4/moments/SquaredDifferenceSquaredDifferencesiamese/scala4/Add#siamese/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
1siamese/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4/moments/varianceMean(siamese/scala4/moments/SquaredDifference1siamese/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
siamese/scala4/moments/SqueezeSqueezesiamese/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
 siamese/scala4/moments/Squeeze_1Squeezesiamese/scala4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
>siamese/scala4/siamese/scala4/bn/moving_mean/local_step/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepIsiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
<siamese/scala4/siamese/scala4/bn/moving_mean/local_step/readIdentity7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readsiamese/scala4/moments/Squeeze*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMul@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub$siamese/scala4/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
isiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biased@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Lsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepLsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x$siamese/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
siamese/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/AssignAssign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepMsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
@siamese/scala4/siamese/scala4/bn/moving_variance/local_step/readIdentity;siamese/scala4/siamese/scala4/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read siamese/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub&siamese/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
Lsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepRsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Gsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x&siamese/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
siamese/scala4/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala4/cond/Switch_1Switchsiamese/scala4/moments/Squeezesiamese/scala4/cond/pred_id*1
_class'
%#loc:@siamese/scala4/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese/scala4/cond/Switch_2Switch siamese/scala4/moments/Squeeze_1siamese/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese/scala4/cond/MergeMergesiamese/scala4/cond/Switch_3siamese/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
siamese/scala4/batchnorm/RsqrtRsqrtsiamese/scala4/batchnorm/add*
T0*
_output_shapes	
:�
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
siamese/scala4/ReluRelusiamese/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
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
Hsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala5/conv/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:��*

seed*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
seed2�
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
<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala5/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: 
�
6siamese/scala5/conv/weights/Regularizer/l2_regularizerMul<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2Loss*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: *
T0
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
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
_output_shapes	
:�
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
siamese/scala5/splitSplitsiamese/scala5/split/split_dimsiamese/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
X
siamese/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/split_1Split siamese/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
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
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
\
siamese/scala5/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala5/concatConcatV2siamese/scala5/Conv2Dsiamese/scala5/Conv2D_1siamese/scala5/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese/scala5/AddAddsiamese/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
�
siamese/scala1_1/Conv2DConv2DPlaceholder_3 siamese/scala1/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`
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
siamese/scala1_1/moments/meanMeansiamese/scala1_1/Add/siamese/scala1_1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala1_1/moments/StopGradientStopGradientsiamese/scala1_1/moments/mean*
T0*&
_output_shapes
:`
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
 siamese/scala1_1/moments/SqueezeSqueezesiamese/scala1_1/moments/mean*
T0*
_output_shapes
:`*
squeeze_dims
 
�
"siamese/scala1_1/moments/Squeeze_1Squeeze!siamese/scala1_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
&siamese/scala1_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese/scala1_1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese/scala1_1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese/scala1_1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese/scala1_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
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
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese/scala1_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
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
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
g
siamese/scala1_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala1_1/cond/switch_tIdentitysiamese/scala1_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala1_1/cond/switch_fIdentitysiamese/scala1_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala1_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala1_1/cond/Switch_1Switch siamese/scala1_1/moments/Squeezesiamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese/scala1_1/moments/Squeeze
�
siamese/scala1_1/cond/Switch_2Switch"siamese/scala1_1/moments/Squeeze_1siamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese/scala1_1/moments/Squeeze_1
�
siamese/scala1_1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/MergeMergesiamese/scala1_1/cond/Switch_3 siamese/scala1_1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
�
siamese/scala1_1/cond/Merge_1Mergesiamese/scala1_1/cond/Switch_4 siamese/scala1_1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
e
 siamese/scala1_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese/scala1_1/batchnorm/addAddsiamese/scala1_1/cond/Merge_1 siamese/scala1_1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese/scala1_1/batchnorm/RsqrtRsqrtsiamese/scala1_1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese/scala1_1/batchnorm/mulMul siamese/scala1_1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese/scala1_1/batchnorm/mul_1Mulsiamese/scala1_1/Addsiamese/scala1_1/batchnorm/mul*&
_output_shapes
:{{`*
T0
�
 siamese/scala1_1/batchnorm/mul_2Mulsiamese/scala1_1/cond/Mergesiamese/scala1_1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese/scala1_1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese/scala1_1/batchnorm/mul_2*
T0*
_output_shapes
:`
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
siamese/scala2_1/ConstConst*
_output_shapes
: *
value	B :*
dtype0
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
"siamese/scala2_1/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese/scala2_1/Conv2D_1Conv2Dsiamese/scala2_1/split:1siamese/scala2_1/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0
^
siamese/scala2_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/concatConcatV2siamese/scala2_1/Conv2Dsiamese/scala2_1/Conv2D_1siamese/scala2_1/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
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
siamese/scala2_1/moments/meanMeansiamese/scala2_1/Add/siamese/scala2_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese/scala2_1/moments/StopGradientStopGradientsiamese/scala2_1/moments/mean*'
_output_shapes
:�*
T0
�
*siamese/scala2_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala2_1/Add%siamese/scala2_1/moments/StopGradient*
T0*'
_output_shapes
:99�
�
3siamese/scala2_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese/scala2_1/moments/varianceMean*siamese/scala2_1/moments/SquaredDifference3siamese/scala2_1/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala2_1/moments/SqueezeSqueezesiamese/scala2_1/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese/scala2_1/moments/Squeeze_1Squeeze!siamese/scala2_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese/scala2_1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala2_1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese/scala2_1/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese/scala2_1/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
Isiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese/scala2_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala2_1/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
g
siamese/scala2_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala2_1/cond/switch_tIdentitysiamese/scala2_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala2_1/cond/switch_fIdentitysiamese/scala2_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala2_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese/scala2_1/cond/Switch_2Switch"siamese/scala2_1/moments/Squeeze_1siamese/scala2_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala2_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
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
 siamese/scala2_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
siamese/scala2_1/batchnorm/subSubsiamese/scala2/bn/beta/read siamese/scala2_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese/scala2_1/batchnorm/add_1Add siamese/scala2_1/batchnorm/mul_1siamese/scala2_1/batchnorm/sub*'
_output_shapes
:99�*
T0
q
siamese/scala2_1/ReluRelu siamese/scala2_1/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese/scala2_1/poll/MaxPoolMaxPoolsiamese/scala2_1/Relu*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides

�
siamese/scala3_1/Conv2DConv2Dsiamese/scala2_1/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese/scala3_1/AddAddsiamese/scala3_1/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese/scala3_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
*siamese/scala3_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala3_1/Add%siamese/scala3_1/moments/StopGradient*
T0*'
_output_shapes
:�
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
 siamese/scala3_1/moments/SqueezeSqueezesiamese/scala3_1/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese/scala3_1/moments/Squeeze_1Squeeze!siamese/scala3_1/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese/scala3_1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Hsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese/scala3_1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese/scala3_1/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala3_1/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
siamese/scala3_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala3_1/cond/Switch_1Switch siamese/scala3_1/moments/Squeezesiamese/scala3_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala3_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_2Switch"siamese/scala3_1/moments/Squeeze_1siamese/scala3_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala3_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/MergeMergesiamese/scala3_1/cond/Switch_3 siamese/scala3_1/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
siamese/scala3_1/batchnorm/addAddsiamese/scala3_1/cond/Merge_1 siamese/scala3_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala3_1/batchnorm/RsqrtRsqrtsiamese/scala3_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala3_1/batchnorm/mulMul siamese/scala3_1/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese/scala3_1/batchnorm/mul_1Mulsiamese/scala3_1/Addsiamese/scala3_1/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese/scala3_1/batchnorm/mul_2Mulsiamese/scala3_1/cond/Mergesiamese/scala3_1/batchnorm/mul*
T0*
_output_shapes	
:�
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
siamese/scala3_1/ReluRelu siamese/scala3_1/batchnorm/add_1*'
_output_shapes
:�*
T0
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
siamese/scala4_1/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese/scala4_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/split_1Split"siamese/scala4_1/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala4_1/Conv2DConv2Dsiamese/scala4_1/splitsiamese/scala4_1/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese/scala4_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/concatConcatV2siamese/scala4_1/Conv2Dsiamese/scala4_1/Conv2D_1siamese/scala4_1/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala4_1/AddAddsiamese/scala4_1/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese/scala4_1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
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
*siamese/scala4_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala4_1/Add%siamese/scala4_1/moments/StopGradient*
T0*'
_output_shapes
:�
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
 siamese/scala4_1/moments/SqueezeSqueezesiamese/scala4_1/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese/scala4_1/moments/Squeeze_1Squeeze!siamese/scala4_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese/scala4_1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese/scala4_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese/scala4_1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese/scala4_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese/scala4_1/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Isiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese/scala4_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
g
siamese/scala4_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala4_1/cond/switch_tIdentitysiamese/scala4_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala4_1/cond/switch_fIdentitysiamese/scala4_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala4_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
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
siamese/scala4_1/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese/scala4_1/cond/Merge_1Mergesiamese/scala4_1/cond/Switch_4 siamese/scala4_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala4_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese/scala4_1/batchnorm/addAddsiamese/scala4_1/cond/Merge_1 siamese/scala4_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala4_1/batchnorm/RsqrtRsqrtsiamese/scala4_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala4_1/batchnorm/mulMul siamese/scala4_1/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
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
 siamese/scala4_1/batchnorm/add_1Add siamese/scala4_1/batchnorm/mul_1siamese/scala4_1/batchnorm/sub*'
_output_shapes
:�*
T0
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
siamese/scala5_1/splitSplit siamese/scala5_1/split/split_dimsiamese/scala4_1/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese/scala5_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala5_1/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese/scala5_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/concatConcatV2siamese/scala5_1/Conv2Dsiamese/scala5_1/Conv2D_1siamese/scala5_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
score/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
score/splitSplitscore/split/split_dimscore/transpose*
T0*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split
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
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*
T0*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split
�
score/Conv2DConv2Dscore/split_1score/split*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0
�
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*
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
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*&
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
score/Conv2D_6Conv2Dscore/split_1:6score/split:6*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
score/Conv2D_7Conv2Dscore/split_1:7score/split:7*
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
S
score/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
score/concatConcatV2score/Conv2Dscore/Conv2D_1score/Conv2D_2score/Conv2D_3score/Conv2D_4score/Conv2D_5score/Conv2D_6score/Conv2D_7score/concat/axis*&
_output_shapes
:*

Tidx0*
T0*
N
o
score/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose_1	Transposescore/concatscore/transpose_1/perm*
Tperm0*
T0*&
_output_shapes
:
�
 adjust/weights/Initializer/ConstConst*&
_output_shapes
:*!
_class
loc:@adjust/weights*%
valueB*o�:*
dtype0
�
adjust/weights
VariableV2*
shared_name *!
_class
loc:@adjust/weights*
	container *
shape:*
dtype0*&
_output_shapes
:
�
adjust/weights/AssignAssignadjust/weights adjust/weights/Initializer/Const*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
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
0adjust/weights/Regularizer/l2_regularizer/L2LossL2Lossadjust/weights/read*
_output_shapes
: *
T0*!
_class
loc:@adjust/weights
�
)adjust/weights/Regularizer/l2_regularizerMul/adjust/weights/Regularizer/l2_regularizer/scale0adjust/weights/Regularizer/l2_regularizer/L2Loss*
T0*!
_class
loc:@adjust/weights*
_output_shapes
: 
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
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@adjust/biases*
	container *
shape:
�
adjust/biases/AssignAssignadjust/biasesadjust/biases/Initializer/Const* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
adjust/biases/readIdentityadjust/biases*
_output_shapes
:*
T0* 
_class
loc:@adjust/biases
�
.adjust/biases/Regularizer/l2_regularizer/scaleConst* 
_class
loc:@adjust/biases*
valueB
 *o:*
dtype0*
_output_shapes
: 
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
adjust/Conv2DConv2Dscore/transpose_1adjust/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
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
save/AssignAssignadjust/biasessave/RestoreV2*
use_locking(*
T0* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignadjust/weightssave/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
�
save/Assign_2Assignsiamese/scala1/bn/betasave/RestoreV2:2*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(
�
save/Assign_3Assignsiamese/scala1/bn/gammasave/RestoreV2:3*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`
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
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2:5*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
save/Assign_6Assignsiamese/scala1/conv/biasessave/RestoreV2:6*
_output_shapes
:`*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(
�
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2:7*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`
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
save/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave/RestoreV2:9*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave/RestoreV2:10*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(
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
save/Assign_12Assignsiamese/scala2/bn/betasave/RestoreV2:12*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta
�
save/Assign_13Assignsiamese/scala2/bn/gammasave/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�
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
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2:16*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
�
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2:17*
validate_shape(*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
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
save/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave/RestoreV2:19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2:23*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2:24*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_27Assignsiamese/scala3/conv/weightssave/RestoreV2:27*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
save/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave/RestoreV2:28*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave/RestoreV2:29*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2:31*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2:32*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta
�
save/Assign_33Assignsiamese/scala4/bn/gammasave/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2:34*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
save/Assign_35Assign!siamese/scala4/bn/moving_variancesave/RestoreV2:35*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(
�
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave/RestoreV2:41*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2:42*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases
�
save/Assign_43Assignsiamese/scala5/conv/weightssave/RestoreV2:43*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*
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
siamese_1/scala1/AddAddsiamese_1/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:;;`
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
%siamese_1/scala1/moments/StopGradientStopGradientsiamese_1/scala1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese_1/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
3siamese_1/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala1/moments/varianceMean*siamese_1/scala1/moments/SquaredDifference3siamese_1/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_1/scala1/moments/SqueezeSqueezesiamese_1/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
�
"siamese_1/scala1/moments/Squeeze_1Squeeze!siamese_1/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_1/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Nsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_1/scala1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_1/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_1/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_1/scala1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
usiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
�
Tsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
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
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
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
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_1/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
c
siamese_1/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala1/cond/switch_tIdentitysiamese_1/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala1/cond/switch_fIdentitysiamese_1/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala1/cond/Switch_1Switch siamese_1/scala1/moments/Squeezesiamese_1/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese_1/scala1/moments/Squeeze
�
siamese_1/scala1/cond/Switch_2Switch"siamese_1/scala1/moments/Squeeze_1siamese_1/scala1/cond/pred_id*5
_class+
)'loc:@siamese_1/scala1/moments/Squeeze_1* 
_output_shapes
:`:`*
T0
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
N*
_output_shapes

:`: *
T0
�
siamese_1/scala1/cond/Merge_1Mergesiamese_1/scala1/cond/Switch_4 siamese_1/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_1/scala1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
 siamese_1/scala1/batchnorm/mul_1Mulsiamese_1/scala1/Addsiamese_1/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
 siamese_1/scala1/batchnorm/mul_2Mulsiamese_1/scala1/cond/Mergesiamese_1/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_1/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_1/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
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
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:`*
T0
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
siamese_1/scala2/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_1/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/split_1Split"siamese_1/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
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
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
^
siamese_1/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
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
siamese_1/scala2/moments/meanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala2/Add%siamese_1/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
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
 siamese_1/scala2/moments/SqueezeSqueezesiamese_1/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_1/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
(siamese_1/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_1/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
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
siamese_1/scala2/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala2/cond/Switch_1Switch siamese_1/scala2/moments/Squeezesiamese_1/scala2/cond/pred_id*3
_class)
'%loc:@siamese_1/scala2/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_1/scala2/cond/Switch_2Switch"siamese_1/scala2/moments/Squeeze_1siamese_1/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_1/scala2/moments/Squeeze_1
�
siamese_1/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_1/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_1/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/MergeMergesiamese_1/scala2/cond/Switch_3 siamese_1/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala2/cond/Merge_1Mergesiamese_1/scala2/cond/Switch_4 siamese_1/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/batchnorm/addAddsiamese_1/scala2/cond/Merge_1 siamese_1/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala2/batchnorm/RsqrtRsqrtsiamese_1/scala2/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_1/scala2/batchnorm/mulMul siamese_1/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_1/scala2/batchnorm/mul_1Mulsiamese_1/scala2/Addsiamese_1/scala2/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_1/scala2/batchnorm/mul_2Mulsiamese_1/scala2/cond/Mergesiamese_1/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_1/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
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
siamese_1/scala2/poll/MaxPoolMaxPoolsiamese_1/scala2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
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
siamese_1/scala3/moments/meanMeansiamese_1/scala3/Add/siamese_1/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_1/scala3/moments/StopGradientStopGradientsiamese_1/scala3/moments/mean*
T0*'
_output_shapes
:�
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
!siamese_1/scala3/moments/varianceMean*siamese_1/scala3/moments/SquaredDifference3siamese_1/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_1/scala3/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Nsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese_1/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_1/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
siamese_1/scala3/cond/switch_tIdentitysiamese_1/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese_1/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_1/scala3/cond/pred_id*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_1/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_1/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/MergeMergesiamese_1/scala3/cond/Switch_3 siamese_1/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_1/scala3/cond/Merge_1Mergesiamese_1/scala3/cond/Switch_4 siamese_1/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_1/scala3/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_1/scala3/batchnorm/addAddsiamese_1/scala3/cond/Merge_1 siamese_1/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
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
 siamese_1/scala3/batchnorm/mul_2Mulsiamese_1/scala3/cond/Mergesiamese_1/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
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
siamese_1/scala4/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_1/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_1/scala4/split_1Split"siamese_1/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
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
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_1/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/concatConcatV2siamese_1/scala4/Conv2Dsiamese_1/scala4/Conv2D_1siamese_1/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_1/scala4/AddAddsiamese_1/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_1/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala4/moments/meanMeansiamese_1/scala4/Add/siamese_1/scala4/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
 siamese_1/scala4/moments/SqueezeSqueezesiamese_1/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_1/scala4/moments/Squeeze_1Squeeze!siamese_1/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_1/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0
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
Nsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_1/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
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
 siamese_1/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_1/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_1/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
"siamese_1/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
siamese_1/scala4/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
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
siamese_1/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_1/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_1/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/MergeMergesiamese_1/scala4/cond/Switch_3 siamese_1/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala4/cond/Merge_1Mergesiamese_1/scala4/cond/Switch_4 siamese_1/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/batchnorm/addAddsiamese_1/scala4/cond/Merge_1 siamese_1/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala4/batchnorm/RsqrtRsqrtsiamese_1/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/mulMul siamese_1/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_1/scala4/batchnorm/mul_1Mulsiamese_1/scala4/Addsiamese_1/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/add_1Add siamese_1/scala4/batchnorm/mul_1siamese_1/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
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
"siamese_1/scala5/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_1/scala5/AddAddsiamese_1/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
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
Placeholder_6Placeholder*&
_output_shapes
:*
shape:*
dtype0
r
Placeholder_7Placeholder*(
_output_shapes
:��*
shape:��*
dtype0
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
save_1/SaveV2/shape_and_slicesConst*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesadjust/biasesadjust/weightssiamese/scala1/bn/betasiamese/scala1/bn/gammasiamese/scala1/bn/moving_mean!siamese/scala1/bn/moving_variancesiamese/scala1/conv/biasessiamese/scala1/conv/weights3siamese/scala1/siamese/scala1/bn/moving_mean/biased7siamese/scala1/siamese/scala1/bn/moving_mean/local_step7siamese/scala1/siamese/scala1/bn/moving_variance/biased;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsiamese/scala2/bn/betasiamese/scala2/bn/gammasiamese/scala2/bn/moving_mean!siamese/scala2/bn/moving_variancesiamese/scala2/conv/biasessiamese/scala2/conv/weights3siamese/scala2/siamese/scala2/bn/moving_mean/biased7siamese/scala2/siamese/scala2/bn/moving_mean/local_step7siamese/scala2/siamese/scala2/bn/moving_variance/biased;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsiamese/scala3/bn/betasiamese/scala3/bn/gammasiamese/scala3/bn/moving_mean!siamese/scala3/bn/moving_variancesiamese/scala3/conv/biasessiamese/scala3/conv/weights3siamese/scala3/siamese/scala3/bn/moving_mean/biased7siamese/scala3/siamese/scala3/bn/moving_mean/local_step7siamese/scala3/siamese/scala3/bn/moving_variance/biased;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsiamese/scala4/bn/betasiamese/scala4/bn/gammasiamese/scala4/bn/moving_mean!siamese/scala4/bn/moving_variancesiamese/scala4/conv/biasessiamese/scala4/conv/weights3siamese/scala4/siamese/scala4/bn/moving_mean/biased7siamese/scala4/siamese/scala4/bn/moving_mean/local_step7siamese/scala4/siamese/scala4/bn/moving_variance/biased;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsiamese/scala5/conv/biasessiamese/scala5/conv/weights*:
dtypes0
.2,
�
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const*
_output_shapes
: 
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
save_1/AssignAssignadjust/biasessave_1/RestoreV2*
use_locking(*
T0* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:
�
save_1/Assign_1Assignadjust/weightssave_1/RestoreV2:1*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights
�
save_1/Assign_2Assignsiamese/scala1/bn/betasave_1/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_3Assignsiamese/scala1/bn/gammasave_1/RestoreV2:3*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma
�
save_1/Assign_4Assignsiamese/scala1/bn/moving_meansave_1/RestoreV2:4*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save_1/Assign_5Assign!siamese/scala1/bn/moving_variancesave_1/RestoreV2:5*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_6Assignsiamese/scala1/conv/biasessave_1/RestoreV2:6*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save_1/Assign_7Assignsiamese/scala1/conv/weightssave_1/RestoreV2:7*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(
�
save_1/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave_1/RestoreV2:8*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
save_1/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave_1/RestoreV2:10*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(
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
save_1/Assign_13Assignsiamese/scala2/bn/gammasave_1/RestoreV2:13*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(
�
save_1/Assign_14Assignsiamese/scala2/bn/moving_meansave_1/RestoreV2:14*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
save_1/Assign_15Assign!siamese/scala2/bn/moving_variancesave_1/RestoreV2:15*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_17Assignsiamese/scala2/conv/weightssave_1/RestoreV2:17*
validate_shape(*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
save_1/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave_1/RestoreV2:18*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
save_1/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave_1/RestoreV2:19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
save_1/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave_1/RestoreV2:20*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave_1/RestoreV2:21*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_22Assignsiamese/scala3/bn/betasave_1/RestoreV2:22*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_23Assignsiamese/scala3/bn/gammasave_1/RestoreV2:23*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_25Assign!siamese/scala3/bn/moving_variancesave_1/RestoreV2:25*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
save_1/Assign_26Assignsiamese/scala3/conv/biasessave_1/RestoreV2:26*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases
�
save_1/Assign_27Assignsiamese/scala3/conv/weightssave_1/RestoreV2:27*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
save_1/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave_1/RestoreV2:28*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_32Assignsiamese/scala4/bn/betasave_1/RestoreV2:32*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(
�
save_1/Assign_33Assignsiamese/scala4/bn/gammasave_1/RestoreV2:33**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save_1/Assign_35Assign!siamese/scala4/bn/moving_variancesave_1/RestoreV2:35*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save_1/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave_1/RestoreV2:38*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave_1/RestoreV2:40*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_42Assignsiamese/scala5/conv/biasessave_1/RestoreV2:42*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases
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
siamese_2/scala1/Conv2DConv2DPlaceholder_4 siamese/scala1/conv/weights/read*&
_output_shapes
:;;`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
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
siamese_2/scala1/moments/meanMeansiamese_2/scala1/Add/siamese_2/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala1/moments/StopGradientStopGradientsiamese_2/scala1/moments/mean*
T0*&
_output_shapes
:`
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
!siamese_2/scala1/moments/varianceMean*siamese_2/scala1/moments/SquaredDifference3siamese_2/scala1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
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
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_2/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_2/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
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
usiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_2/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese_2/scala1/cond/switch_fIdentitysiamese_2/scala1/cond/Switch*
_output_shapes
: *
T0

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
siamese_2/scala1/cond/Switch_2Switch"siamese_2/scala1/moments/Squeeze_1siamese_2/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese_2/scala1/moments/Squeeze_1
�
siamese_2/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
 siamese_2/scala1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_2/scala1/batchnorm/addAddsiamese_2/scala1/cond/Merge_1 siamese_2/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
n
 siamese_2/scala1/batchnorm/RsqrtRsqrtsiamese_2/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_2/scala1/batchnorm/mulMul siamese_2/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese_2/scala1/batchnorm/mul_1Mulsiamese_2/scala1/Addsiamese_2/scala1/batchnorm/mul*&
_output_shapes
:;;`*
T0
�
 siamese_2/scala1/batchnorm/mul_2Mulsiamese_2/scala1/cond/Mergesiamese_2/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_2/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_2/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese_2/scala1/batchnorm/add_1Add siamese_2/scala1/batchnorm/mul_1siamese_2/scala1/batchnorm/sub*&
_output_shapes
:;;`*
T0
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
 siamese_2/scala2/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala2/splitSplit siamese_2/scala2/split/split_dimsiamese_2/scala1/poll/MaxPool*8
_output_shapes&
$:0:0*
	num_split*
T0
Z
siamese_2/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala2/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese_2/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_2/scala2/AddAddsiamese_2/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_2/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala2/moments/meanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
!siamese_2/scala2/moments/varianceMean*siamese_2/scala2/moments/SquaredDifference3siamese_2/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
&siamese_2/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_2/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_2/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_2/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_2/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_2/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Tsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese_2/scala2/cond/switch_fIdentitysiamese_2/scala2/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese_2/scala2/cond/pred_idIdentityis_training_2*
T0
*
_output_shapes
: 
�
siamese_2/scala2/cond/Switch_1Switch siamese_2/scala2/moments/Squeezesiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_2/scala2/moments/Squeeze
�
siamese_2/scala2/cond/Switch_2Switch"siamese_2/scala2/moments/Squeeze_1siamese_2/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_2/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
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
siamese_2/scala2/batchnorm/addAddsiamese_2/scala2/cond/Merge_1 siamese_2/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
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
 siamese_2/scala2/batchnorm/mul_1Mulsiamese_2/scala2/Addsiamese_2/scala2/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_2/scala2/batchnorm/mul_2Mulsiamese_2/scala2/cond/Mergesiamese_2/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_2/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_2/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala2/batchnorm/add_1Add siamese_2/scala2/batchnorm/mul_1siamese_2/scala2/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_2/scala2/ReluRelu siamese_2/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese_2/scala2/poll/MaxPoolMaxPoolsiamese_2/scala2/Relu*
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
siamese_2/scala3/Conv2DConv2Dsiamese_2/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�
�
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
/siamese_2/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala3/moments/meanMeansiamese_2/scala3/Add/siamese_2/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
�
3siamese_2/scala3/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_2/scala3/moments/varianceMean*siamese_2/scala3/moments/SquaredDifference3siamese_2/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_2/scala3/moments/SqueezeSqueezesiamese_2/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_2/scala3/moments/Squeeze_1Squeeze!siamese_2/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_2/scala3/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_2/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
ksiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_2/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese_2/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_2/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_2/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_2/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
siamese_2/scala3/cond/SwitchSwitchis_training_2is_training_2*
T0
*
_output_shapes
: : 
k
siamese_2/scala3/cond/switch_tIdentitysiamese_2/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese_2/scala3/cond/Switch_2Switch"siamese_2/scala3/moments/Squeeze_1siamese_2/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_2/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/MergeMergesiamese_2/scala3/cond/Switch_3 siamese_2/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_2/scala3/cond/Merge_1Mergesiamese_2/scala3/cond/Switch_4 siamese_2/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala3/batchnorm/addAddsiamese_2/scala3/cond/Merge_1 siamese_2/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_2/scala3/batchnorm/RsqrtRsqrtsiamese_2/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/mulMul siamese_2/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_1Mulsiamese_2/scala3/Addsiamese_2/scala3/batchnorm/mul*
T0*'
_output_shapes
:

�
�
 siamese_2/scala3/batchnorm/mul_2Mulsiamese_2/scala3/cond/Mergesiamese_2/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_2/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_2/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala3/batchnorm/add_1Add siamese_2/scala3/batchnorm/mul_1siamese_2/scala3/batchnorm/sub*'
_output_shapes
:

�*
T0
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
siamese_2/scala4/splitSplit siamese_2/scala4/split/split_dimsiamese_2/scala3/Relu*:
_output_shapes(
&:

�:

�*
	num_split*
T0
Z
siamese_2/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala4/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala4/split_1Split"siamese_2/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
^
siamese_2/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_2/scala4/AddAddsiamese_2/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_2/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala4/moments/meanMeansiamese_2/scala4/Add/siamese_2/scala4/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
 siamese_2/scala4/moments/SqueezeSqueezesiamese_2/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_2/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_2/scala4/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_2/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_2/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_2/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_2/scala4/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_2/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_2/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
g
siamese_2/scala4/cond/SwitchSwitchis_training_2is_training_2*
T0
*
_output_shapes
: : 
k
siamese_2/scala4/cond/switch_tIdentitysiamese_2/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala4/cond/switch_fIdentitysiamese_2/scala4/cond/Switch*
_output_shapes
: *
T0

Y
siamese_2/scala4/cond/pred_idIdentityis_training_2*
_output_shapes
: *
T0

�
siamese_2/scala4/cond/Switch_1Switch siamese_2/scala4/moments/Squeezesiamese_2/scala4/cond/pred_id*3
_class)
'%loc:@siamese_2/scala4/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_2/scala4/cond/Switch_2Switch"siamese_2/scala4/moments/Squeeze_1siamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala4/moments/Squeeze_1
�
siamese_2/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_2/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_2/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_2/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/MergeMergesiamese_2/scala4/cond/Switch_3 siamese_2/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
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
 siamese_2/scala4/batchnorm/RsqrtRsqrtsiamese_2/scala4/batchnorm/add*
T0*
_output_shapes	
:�
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
 siamese_2/scala4/batchnorm/mul_2Mulsiamese_2/scala4/cond/Mergesiamese_2/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_2/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala4/batchnorm/add_1Add siamese_2/scala4/batchnorm/mul_1siamese_2/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_2/scala5/ConstConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_2/scala5/split_1Split"siamese_2/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_2/scala5/Conv2DConv2Dsiamese_2/scala5/splitsiamese_2/scala5/split_1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

^
siamese_2/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/concatConcatV2siamese_2/scala5/Conv2Dsiamese_2/scala5/Conv2D_1siamese_2/scala5/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
�
ConstConst*��
value��B���"���E-�,�׽�i���ܺ����;=����z��=��<�*<�J�<����ɢ�@��::�d�W䊽+ɽ��׽ ��;��>v���*m����`E�=� ɭ<`= f׻�j�O-���H޽��@��%��q=�,���;�>�8�3�0��*���M�òν��X�;��6=�'ۼ(���t�ͼY;<@���~.���ɽXy��g���آ<�/=�Z��ҭ�=b�!�{X����=,W�� �n;(3��귽�wO�����,ȕ<t���"�=U����0=�L�=A:��ڻD-�<�v����;<|aܼ��{����[9��O�=�fƽ�6P=��6=
2}=K�$=%�������=��
�{W��8/h�8�x�����g���x_��н΄��ˑ�� d;ွ��<آ���P@n�H�Ƽ����i*�}�=���Փ��������șw<H락'ד�_헽酟=�������؋@�Ļ�`A';���� <�8���==�9=�= �{<�d��hT����O�2���<d_*�
:	�^�f������{x��G/�����X}<�2�8E�?��=�8����=���<�o8���<�w���>=�M�< &~��:=��漖2= �;��|=r�=�>=`�<������=XF���.�No<��l�`�F=b�I=���;8C<o���ۿ<Ⱥ<�؛��)�� ����;B9V��-�=���<��|��<J>�|����cC=���;Q<m��=�P�<�׼cۼ�~�<a�(=��=��˻&9�=��<@j��x�>L�=@#�6<����!�=��t=4��<Tɽ`�Z��FR= F����2<�D|<@07;��<&=(� ��:��q�Cݏ=�0���o=���:te�<I�<��̼ ��8xYD<hOL���.<���;��<@�@�h��<��\����Ȫ�<ԩ������tmǼ�`���<02�<�_=���=@���@KU�DD=�Ţ;��S<4��<ȩs����.YĽ�s�;�/ȼ䐵�n�=|���c�=`C�|������<����0w�<�.���R�ʄY��)��h���[�=�ى��l���]�T��<T���԰�<�A=
����|����q�����`7��9۽ ���V52���<>:��$����t
;�[K���;�.���ɽ�'�@�^;&8�
�%���]�tb�м�PdZ� ��4�0�;L���W;��ƽE��=P���߽"��=��<h
��'������#s�������֯<�)��0M�;�f��}=`���8���� �89�NE���<�I�x�,�����s�����=rm�~�� �u� �A���0=Ib�0̽�G�<g�޽1���8����=����m��A���q��d˼�������Σ� ��;�k
�F���^����Y��k
��f������2N�~�(� �E;�B��`e�; ���=�Ѩ���ÿ�QSȽ��ӽ�A�Sk�p*�����z������<܍�<��+=0�j<H�U���;�B��߭�p�ƻ����@�l`��X�%��A<`G�<I�dB�<�,L�(A&�]�R=�M=$�u=O=o�x)=�'�1�=�F�:�}��+d:=����=,�<<��<ћ.=Ts�<`<�;�-w���= :/:�����;0�	���2=U>g=��T��G<8�%��M=���;���;��ںT���< ��(��<�<Z��`�8<�� �Ƞc�^�=�K;���<Ld= w3;4L�����h�=\��<���<0�t<��B=Ԝ�<('
����}V�=�;0�㼨2a�� N=^PA=PJ�<�tt�,���R<`���e˻��t<(`�< �b9���<�e���5<|[����|=��4��=p��;�7=�b�;��2����;<~�<�!�D��< (�; ϖ<����3�<L����䲼X,=`�A��Dں [X�����5�;r�%=�-�<�,�=��m���\��*�<plY<H�<�<@AI� �������66<P��m��я�=�ӽz`�=P��<0��;�=<�� �n�:�\)h=�I���V��HyM<p�ݻ춑��ͩ=t�E��2����>���>��⛽�-U=�|�<h���Й!��^��.l����<��d�ະ�h'����=_D��Ϙ�PR�����X�<��I�Ƚ��@��A�]�����t�XaȽ�q����<��j=0RA�O�0=����D<�콖jb=8C̽2B?���<��B=�׻�C��b*������ýt��=������<���<>]=>��k��T��<��=H	X�FI=W�0
����L�.�\�~�;�"꽕Ľ�D����[��i�<h���z��3��
��қ��Oʼ<�=$ּbA�� <О�|��<x��H�@�`�P� _�<�8F=�畽dg���$ҽ|D��x�R�H��b��.���"i=W����b�<0�U�&�� _�u^��QR��F�	���NU�l�"� Tκ���bR� \���S��S�0=\�<|��$��<bH�P�Ļ ��;xX̼t��< �O< T;(��<R�H=`�G<�2h;B�n�@��;�V= �=/JQ=���<��w�J= ~K�t��<�� � �Z:��<H)r<�!�<�u�<�$�|t�<ĩ�<@�};���³=���<@��;� ;�=��J=G�V=d����G<H�^�l�= �9;�=��,<|���ds�<�B<`&D��� <�si�-�=�2��oA���=��M;��,=d�<�l��@��(�<d�<��y:�{"<��<�:�<l��< �����<�۟=�< F$;@+����<\`�<�̏<��~�и{���N����״�č�<q�=�$�`�=Tļh��< _8�3�V=i��H��<������<�����X:����#=��6���-=���;��<�׫�$y�<��6<�����D�<}���|[<��<K����))V=�8Z�L��<(8��8�� |#:�<���:x�< �ջ�<2���E��k=XCk�l�P�4Ո=V�˽�-�=
�h=��=~�=�V��̒<�Q��=���:)����$=}�2=�0���s�=��/�7|��� ��Ȟ���q�C=L��<'��@.z�`����ܽ��'=м���% <�kļ��=@�Q���ǽ��������T=����ڽ�v�ܺ����߼�1ֽR/���ýV�S=��=ح�=�C��||�=,��+�V=��ͽ1��=�G� 󠹰����?=�=���Rۉ=^n �z�ֽ��=�����Y�= �P=� :=�Ur��9H��N�=�:=�rV���v=���-Ľ�=�g�����zߙ�O6�� x� 9�Q^<rgýK����ҼZ�2�x�ȼ��]��&�=�M�;��m�<mq/=��=`�����>��%��G=�
e=��4� �������<\\�£T����=��̼�`�=��4�l��<(H<������$�2�U�r�N�� ��5޽��
��
"�po�;FE�b���Z<@̃:ߍ=���<d����1
<�����Ѽ  �<�$�(~u<ċ��00/�8�<,G= �P��;�X���w�3*o=J��=��=Fh7=,s��7q]=0g�.:=��?� ����<P���v�&=xn<��<��=�I�<��;u����0=��6<x��0��;X��<��k=4�Q=�ue��!A�Ȁ���=�J�;xHJ<���;p�a� *<���؜e<��.< �W�l�<t�L�2���[�=�4#�K�H=p�j=�&����P���< �N:4j�<�c�<�Q�<0,�<@��� �Y�0f�=PD�;,��������=��X=�G=�I6��$];�Q�0��c��dO�<h)7<`4����4=p]t��<�>����y=`�v3=|����>�<�½������B�3"T=�����E=�5S< Y�;p��P��<X�u��Q���==�^ڼ�t<@x�;L����#�:=hC�<��@=�������� �h<�;�غ<t��<D���� �`���{&=t�b�l�Lx4=����=�zH=�D�=.RL=�S�:���~;�=@��y����e=Qt\=  �5U͊= ,�a\��< �ļ�<�=���< ����TH���<�#ּ�O,=Hr<f%8=�t���:�=��%=L"���E���|)��a=p��;"6ڽ$F�<p�Ƽ�	'�eA��  �8m��Y�=�� >���<`��r��= F�<�0�=�Km�|��=�
y;�I�;
����̛=G9=�UνT�>9n�Wý3sm=U�ý�r�=�q=�;�<��Ҽd�˼yU�=yR=D+��Ǖ=0�V��������<�۶�Hp�8Qż$d��P���p��<���<F�����L���|�#�:�[��`�#z>�̗</@�KY=��b=�{�=@E��������6�=<|�<Z$��΅=dp���v=���:Kͼ���=,	����=��w<�X%<!��=T:케�M�vb�B 6���g���}½$ؤ�(�&<H�c�^����u<���{�=P�=�睽 �W��(!�2i����Cm� =�;������<8�=ء����;Q1���A���[= .�=�=�=C�%=\#ۼM�d=�ِ�Y�<X]�(�G��!�<���Y�
=P_�;���<l1�<���<@A�:�Ʌ��R"=�^.� ������:<d�<*c/=:(G=l-��*+��U-����< � ���8<����
����Y;�a�T]�<��C;PsH� QO;�dk����Ά�=���� �=�k=�S��
�Y���X����<XS�@x�<D@<�J�<��9<@(�u�����=�:�f�tk����<�Qo=�=,pk���s;�J�� -��<b¼X�< z�����=D����;?<�;��[G=Hn׼���<0�$�<�<�J��������s�t�5=d[[���@=`U�;H�`�'��"x<P'��hu�v�b=����[:��4�����8v�E�=ȿ�<u�K=$ɼ�pR�I<06%�x|=�M�<���� ���wJ=��=�W� &<&I�=�1��p�%�B9s=;��=�B=J��=�ɬ�$��<@��:p��=���< ��:∪=��?�1���hʋ<d֕<�F�<e[q= ���|5�`O� R�<H�<��=(5F<h�D=xIz<���=���=�����ϼ�O���;�M� ?����9=�u��P��6�V�:-=�_ͼ��I= z >p�M�3��6D=F�=X�=������=�/� ,�B����>��=��q���=K���d�7��H=�����=Uċ=��*�Z�=�Zڼ��	>��=�G�!�=�:��$﫽,﮼��E�B�;=Q�J=t���`Qh;|Ă=���<X�+�,�C�<��<~�A���s�������=ru%�%�ƽ�ۇ<�A�<:�J=(��<�E��p��D�=�f�ړ=8T�=𰟽z-�=��9�X@u<|`==��Լ}�=>`m=�ZI��r�=��<�>����<��;��
��
Y'��S�f�,=(}�<�eܼ.R�� !�9�����S�<3K=B �k9.�8���H��p�u�4Y�<M����x��<Ԥ�<�UǼ@�G;�i��ߠ����]<��=&�)=L��<,NѼ*�9=𔶻�N<^����u< K�:�۔����<�/`�`:��P} < ��:��r��W/�\��<�M�`����Ȼ�G=h�<��=�V�`��`Gu� �4�؛żl��<�ҡ� '��p�:��v���A<xn�����p���ݼ��9��(Z=:��8�<Pt�<D\;�����к��@�����p� �};��:`�m��n�� ��&5�۠z=�q��݃�����%��6v6=�k�<�
*� ZT<jq�,6ϼ��F��'�;,&ϼ�g?��@<��`�<�Z�d��<ĄA��X;\���hT<R��\ ��h�a�(��<���#=��Q�N��p0����V<P� ���ἐ�!=Ľ��0g�;��� &/��o��̜�<�ܻ0E=v����HF������dT=h�<J��Ȝｾㅽ0U�P=M� ���=l_V��YA=��=X
I<��,=~��������<b$'�@ ���������$$�<I
.>���r4&�,k��5�=UW�-��=�
�=Pi`���<�wt�ľ&��]J=����=�,����<��-��L��.�Зu<�nN=��V��~����\<��'=@Є� (�;�?O�ڶ=�HQ����jч�$6��l1�t~/�j)|=��j�	�=���'ޤ���x=����@R<�N��
l�4����W=A'=��.<L Ҽ$�={=��j���K� ӝ<`��Xڕ=^N7�lW�PP)��ռ�/�=�.��X;<�u=��;#=d?���z��fq<�P���#���|=�A�<
�n��d���d���ٽ�%]���E=�g�<l�6�\ս=`p	;lv������v�� E}�T^����J= B��:���O���i��R7	=�&�U$��&�}��1=����Z߼f�W��fr� ���HIa��5��:|;��<�===>!=��2�@⓺`��v��Ǫ<8��@�W��1������0�p<(�X��<�!�@Qɺ�eo=(2/�.�=0��<�t��U�<�GK�+=`�;;�*K����<P����.=PN<�Na=8�!=.�N=��Y< #k�C�;=�rK<Y����=���ל}=�ea=@V
;̰�<��M��	=�C�<�GS<0H�;Ē˼�I�<�%����=�=��l�弘�c�� ����T=�#�<0��<�c�=�H<hoz��t��<�=�O>=F�<����c=~�=�k<!���=��}��ͼ@�s;�:k=�7�=L��<�*��p5k����=`� ��]<pB�<��<�	�<�&=��:�<o7�X]=�A��fV= !-�.<=���<� ���<$8�<DM�z= �<���<ܽ��s	=P����꺈��< +!; ^�:����h�޼�a�<��<k5=Τ�=�v@<�����<�"�;�:=���<�B�,[n���\�h�<��3<�ʪ�|�=�-Ͻ�?.=�>M=�e�<\�|=c��߈ڽ�͘<4���b�&�S�.0"��p�;�R�=h	$�5o���"y�L�Y=_����=s�m=Xa��P�< �M���	�^H=N�ǽܤ=vP���=��[�p�h��k���müҟ=�m�;�a����o��m�<�,v<�%û�{��\���P	r����<^<�����`=⛿�CS=���?�=�^��S���=�Y=�0C;0ʕ<����W���
�����Y�J= �ջ����P �<nL&=������S�@x�:E��`Og����=��(� _F�@��;8A���=bV��^�����<|��D��<��U��\�R0f������������<W��=̼���0o���u*���1�|=�<@��:������_=�i=������� ��f̦�P%��u;�$�;��彊bU=�>��Yv<�l��0��lּ~�_�M7��������/�y:ڽ�z����J���h�ռ�w<��;�q=��(=8r5� ��d)¼����`�m;|5>��Ps<�c޼�7�N<X� =��y;��=��T�(�U���=lڸ<���=��=��XnF=�����?=�����,V�
R=@7���=��O<�T=NcZ=� =X.<h;U��M=�+G<АH��z�<0���q=^ �= ����;#�)4=H��<Dކ< ��;�gʼ���<p�ϼ]�7=�/=�����y�t�y�ܕ�����=���<!�J=P\�= �i:ԗۼ����4-=O�#==�<`�W<9X?=!%=<ʑ<4�Ҽʸ�=�L�;��K���	���9=�K�=�:�<|�xl��ߍi= a����;(kS<��<�;���<����X�<����a=����x@=`�<�V`=�?,<����</=��g#=�ԉ<HI<�K��l��<�J�����ƻx=��R<�P;P�	��Tۼ�<��c=e�=a�= ��:z��@=�/�<y=DИ<Uw���Y�0����<0����*�G=�=�h۽Xi�<۩�=��9=�Z=Y��������<��2�6���?:<�~�<�f}<���=����ɽܷ�����ؓ���=|W=� +��`%=p}\<u~н�/=d����=�̼z؅=����l$��H�D�����	��=�8�< !T�h����X<0�<:���^����Z� 5��r�=-ɶ=�f���N�=��ýԡ=
,��6_�=Pޡ���=�x�<	�< =f���u~�ҡv�8����
�=�LB��<�*uF=�̑<��R�t�.� =0�ƻ|���vt�=t��`oU��]r=�؋���;x4���`m��^�D��@	�Ȭ���3��d�� A���d��i����= º 2D;d�=X��<��s;��숇��!����
=tҹ="��Ԛ����ؚ3� V�����m<����Q�=0L�; �º�-�<�����;{�ν«\�k��'�뽻�׽Q���J�p㻽0����P< �L����=U1G= �G�(}`� Ⲽ\HҼ�!<L�+���<�d�����@`�;�SC=`t�;W�<�]���u���=XB"=lk�=���<P@���qR=�b��D�<Vȼ�������<HV<T��< �b<���<qN/=p��<`�;x��e=8"/< Ҏ���<X��<	�=�\�=X�� HL��Dݼ�=Ծ�<�L<H�C<��@��Ƌ<��ɼ��<�0=�h��(�|<���>���绉=���<৅=�$�= �;;��꼤M����=ԩ�<�<pXi<|��<�[8=�$�<�	�_��=Hٳ<��W;��޻�E=Zi=4ҿ<�sG����/�=�%߻�ûH�m<�#s<,T����<��F�`l�<�s;��i=P[Q��=p�n<�-=���;h�0�`��;/�w=���O#=�Y�; ��:pAp��'g<<����gf<L�m=���;���;�\��@����N8<nZ�=l��<��f=�'��Z'�p��<�*�<�;�<(�k<�\����<x�ͼ���<�n��-g<
�=��m�<�k2=�<�=Ds=�����3/�3xP=p*L���^���<[�E=p�;�A=P��r�� ��:@��p:<8!~�p�<�-��E�<HR�<�'���=��<%�v=����W�?=8�0<�����=Z��09��w�=>��=�'���6�Ph��Do=b�-��1�,�g��=R~&>��=�����4>�"T��;=����$�=���<���e���o�<���<Rp½��=�����gp��1=w����<��P=@��<��a������6=�oռ��½��o=�Z�ht引K�={���^��Nݘ�H��X������¼�H�M5V�����������Y���>�m=p�<O�_=5�=�`N=��I�,�-�B���*;��=�9J���6�<��X�O��2=:�z�"=ʼ/�{��=&n=���V��=�rȽ��<�ʽ�t���5����Խ����8YP����g��)�=����t=d^=0鼽"������Z�L����H�i��rɻ��x�8�|�L>�<�=0^��l��<r�j���P� ��=�3=��=}o~= �;�Y=��j� =��$��v��<е���= �:�b=�E_= =@\f;N��Yzp=�� �0�Լ�ȗ< ��H�=ǘn= E�;hģ�PO�dc=-�<��ļ T��p�� ��=2��K��=� =0Ɲ�@����ͽM~�=H�W<�p�=���=\��<\S��O ��o=�<��]=��j<V=�T=�2�<R,E��=�ݠ<+ļ@/���hY=��=ԁ/=$G��8Me<ޑ=��{; �): t�<�PN��Vڼ|1=@/;X�<	��,�=�>�"S=�=dk=���<lj!�`�;7c�=\�{��=��Z<���;ܪ=��3<��X��U�<\�=��ٻ���\��<�&���<��p= �=�=�5�\'���|C=,X�<:�M=�?�<�R��>u=`�3<8�t<|T����<@�;�H�P���0WG<K~>^�4=`)p<����=���V�:�\b�<�-T=p��X=|�8��� ��9xk��k=�K����;p�Խ�%��$�1=��%=}O9=��<�=�I)<���<��C=ә"���ϼ�w���7�=vH�=����59�m��04�< ֋�@�%;p)�ew�=��c>����׼��:>`��;j$n=�L�á=o�#=4��&T�ON=pä;�V� >�,��,���n�<	�,��<��2=�|<(�������ku=���P���� 4= �b; cܼ�T=͸��傼����`�<�����<vL��*�EWE��/i���{�+����`��>��p=d*񼰴F=M�=�.g=$�<�
t���� ����l'=L�� !�<Y���S<
r=��,��9p<x�޼��>��= �¼�.�=���=o����w����V>�+���Jr��@�d�৛<����-=���,LF=��<8}½�����ü�d�\��[���9���l�,����=$��<ܞּ���<�ld�{ݭ��9�=��=ǔ�=��x=x�B�7Er=:���<dƷ�����v�<t��dc�<@���)�=��G=�^�<�-�t4v�"�:=�/��LR��ખ;��N<q%%=�==��l<�����Q�H��< ��;|xҼ�l-��&���J��1����E=�>�<�Ua�P�K��l����6�= ��:Ǥ]=Ǝ�=��;�u�V��Xq�<��}��yD=䷭<H�<���<or<jw]��L�=e�<��ռ_#���<FЦ=��=�����fG<x �`�;`����=�<d8ϼ��f��	=�X,��h�@Kw;���=�]����= �2=�5�<  շ������;�e|=L�����=п<C��+�����C����<3��=�򝼀}�����������;8�W=^iv=2��=�Uk�8���=��;1V=<�<�7�[o#=F��= B"��5����<r�.=�.h<��]���$=�j�=���<�P~=�˶���;�� ;@�,��<�Y= �A�⯼=��FBf���/�`�l��`�<��<�	���@��	��T=m�s=b�=��`<�9�=�=�&�=|u�=�#���q��O���z<��=D�Z��8<�[�<D��< ��:��<�����=!6<>���Xz���=fL=�>�=�X̻��=h`<� ���IԽ��
>�cG���s�E��=mZֽ�����<Ez��P�f<90L=@o��	QQ=��h���=PB�<��\��c�=𢧻�S����A����]f[=y��=L�<�����=�l�(~p���Ž�㋼M���0佌8� ��=F��ν ã<�Bs<P��<q�4=��8���5�7Y=��x<Ԧ=p�1=L[ͽA?=�A=��(<��i�±%���>$p>�1y�>> �<� l<ܭ��f�c�D�½/��I���ԝ;<Z�<`�J;���!<�?���<,W�<�ÿ��ռ�~�R�J��>)�T?��H�;vV��#��Yv/=؆<Tdм�=�<�|���н��A=�w=(��=�MZ=�pS�#�F= �q;`G�<�tO�@�B;��_;\�⼘��<H@�P0�<zf=����rμ�v�T�<p�� ѱ�P֪�t��<�z,<��=�9<8e���m��㻀�T���?�8���.���hE�.��B� =�kb<P���Bü�X���Y���H=p�L�8
=�eM=l�ɼ�X��t�����:�I��m�<�Y�<@�^��;P���<v�f��y=���;x���	�h�:�㭓=@�<Io�0ȥ<�;(���O����E<R�#��8���q;@!V���M: ��;�=H	����;���<�:�;l�]������;N�&=xp����< !�9���^�0Ż�tҼ��2�,@�=�͢�0��;�1,��*���W�\M�<LF�<��J=�F�`>˼�$��x3ռr:�=���<֣*�l)����׺(�~��\=n���ic= e�(ge�BŒ=0t�;�!J=�_����:����.�1���l<�鸽b\��9"<s0>�T���y�P=6����=ۈa��d�=	Og=��d��&�<�^��-�Ƚ �[=G���T�=0];���M��x��4P��x��� z˹�V�=0���,�����1<�L=�}=�T=ǲ����=4�Ƽ�Ɋ;�g�<s}��s<4���'&D=��	�>d����
�
=�h�;[X=bSӽO�����p<��=D<�av=ƥl�@D<��u=L�=9R}�L߉< ��L�k�v&�=u4���{< ��;dW�<�%�=��<,�<�v~=�h)<0B������`A��T�缜����[�=��<�B`�`�^�R�7�.W��q �����=�6'=(���� �=vp=޻��kF��U��<��@J�;8�=�S<D�$��F�<4�<0J�<�gm<W����R��bA<�i�x�����d*����P�����P�<| �<0��;���<�|=�qT=H����O�<��&���ͼ��<|y���˪< ���vT;�t�<=45�<P �<�v� �|<=��`;�4=p�L<0�¼�#=��`���<8_�\D���9�<|��<�0�< �<pv�<$̶<|�=�LW<��]w%=DA�<�z;���< a*:�|y=��S=�(�h�%<H�~�;�=(e�<�M4=��&<0��;� �<���:��=���<�S�����kt��ly��7-=~�=��=�?J= �\�@7;4��,��<:=�����::�=O�5=��#= K!;6^�=Pg\� ���z*<�e=H+J=첃<�?�xѱ��ґ=���@\5����;�G�<�w�<M�= \;�o�<dڗ�2!=�y]�9�=0�	�r�C=0��;({"<0��;�U�<@�к�_L=�p|<X�S<�,<��=pz���U;��b< ��;�oS<��˻�&,���<��*=Hv<�M�= �<��"��!�;`�t<�ȓ<@�c< \���`<�I���< d=����p?=��������]�j=��=̓�=��d�R�R�V^"�L�ۼ����bH�Zܻ3X<X��=d�������Hާ�L�v=�W佾V=#h`=
��z�	=�=�R�	�==w��b��=t��ܼ�"���l��b&��E,�"��=�[�=i� �Ǻ���<��i=D�=�3�����<�E;Fq6=�n�<�ĩ��L�=��㽛d=(�ڼ /> w���|���V�<�k�����=��Ƚ�޼��m<p��<@Q�; ��<6�m�`�<��<J�(�}j6��i�;�y��Q����=�0��(
 <#�b=���<�^<�ռ��=kf=������pƳ<}%�=����G�� ��<��S=�r�:��=��<<p���Yꆽ�94=�O�<�A߽|�=��.=�8���� ���H�c��:=� =���<��㰋=:T=��<h�-<������<�ߔ�&�$�Ľ������4��׽��Z��(<h�l<`{�� ��:�^/=S�O=(Tp���;`�Լ�D���fi<�ٶ��J�<p�< �;�N����&=�#�<�h<�����r;'J=�i<��,=׬<��¼Z�O=�RZ����<�����t�<ij<`8,;�y<�\H<��<�<P0�;�SǼ"7=�x�<�.;<I�<Զ�<��Z=��M=@͝��&� ��k� =�_�<�6=��%<Pm<�]�<��@��<m&=�"̺�� <Hˉ��rz�>$=��<�[9=b=��_� ʅ��}�r=��<��9���< �<�&=�_= c<N��=��;�5�;P���(��<�1=P�$<��(�~���y=����p���`�;��<p�;@��<��Һ8m<@d�;B�=��� ��<h^�<�.N=�Dq;P�J<�R[<Ni= ���g�B=P��; ϸ�`u;��<���:`t �@G=��1<B�; � ��:��� _<�TW=�2�;C�|=�<I]���}<���<P�S<@��;@��`>=s�;ƀ<I��$=hh�<ȁ��P���:y=_�=< �=�{�����fT���������g�<8)=��<`�f=�M��=W��;�<p眻 :��}�P��<�����L=�K=�8ջ��<ps�����=������Ȼ �o<�tL���
�Z�o��=h��=�Ƀ;`$�;�D4<j��=h�*<�0d�   6$�<���= Ӹ=���">D4ǽ���<��$y�=�K�=�]	�@��0S�a&�=��!�S= ����A�<Ȓ<D�ռ�޼X=L���Y���q۽���<��:�����= b;�X�)<�v�=��(<�䬼��5�`�=pL�<����?���[=��)�{�����D�{J����H$�=L�&=�w=0�H=�ʌ=(�]�ث<�c׻�sȽ�i��6NB=�e9�ĸq�@1�g�;�_7=��S;`�<h˽�4�=���= |�~�=������w='1Ͻ,w���轴
�����r<ɽk���9�<`�+;P��;�i<��3b=�`=
�%�� �ͼ���(<Hۼ��;���`M����B�y =��]<H�+<\�����w= AĺA�E=1�<��$��W*=�.����<$�����K���< ��8����qW�<Ӳ<���<���<@�y;���QQG= .i;����q=�<	a=��	=pU���'���(׼���<�Q�<4�<��d<��<hn�<�1����=j�=@�x�<&^��h�� �=��<�U=�m�=`1� 48�,���x��<h̗< 0�7�y;��<̉"=��Z=`��H�=0D<�%{��/�8^�<C�=0��<h}��nA��y�=���p�����<Z�< ��:� =�J�;@���xy��lD=����g=��<w�=X�L<`�
����;�(=�k.�=�b3��_����Ќ:<<� ���a:r�=��1;��Z���k�P���Rc<3�f=H��<�[=�x<�i�\��<��<�� <�*<�lp�?r�=��0�J<�oK�߉�=�[ͼ T�H�7���(���=��i= R;�
ڽ�^y�?,�������=׫5=�yۻ 	ӻ�ڼ�	߽p��<��A��?�=�ý�;��۽N�=lnw=��U= �9�c:=x��=h�\��$���	=�۽���u+�����=U+�=��Ƽ(�ༀ���h=@}D;8{��`��]�=W7 >�}n=��#��D>,M� ��<�(Y��\@=]��=6-� ^D�P_ἰ�<T���I�=�V����i���漿��`�)�Xkp<��:���~�𥅽��<D��������~��д�;�=`�=�����<��촋�8Շ=��; `����D0R=�}\����א��ĺ�y����@�=6�=�m=�,}=��=��=��:�B��)���V��~� =��G�@�����ƽ̵��p�=z�S�Wz���㼼c�=�E>p��35�=�>l����=ɛܽ��\���`��6�G�T����x���[�=t�>��s9=��i���X= �=ty��f@P�dם��R�G伀�H�n��>M|�X⼀�y� ��;�δ�@��<r���!Q�Jm�=��b;�L�=��=�b<�)�<Za�n�=��ǻ�Mc�r�=���8<}��*<o=�3=lj= �1l��L=��Ҽ������<؄ڼ)�T=?�< �e<������N�T��<���<|e� �.9��<@�x;{�����=�	=�*������U��ZfʽEC=�)�;�4= ��=��Z<��
���X�$��<� �<�(W=���;���<�׸<b�
=�VN���=4ϑ<�����,J5=�Ê=`=���� 嶻ܙ{=P��;`�7;`�<�a �����=�Xw<�:B�����m��=��E��D=�\=@�=�[�<�m� #�;r�=d�M�\>�<0ǒ� �;�Y���ݻ8:�� ���jn=O,�xk����`em���5<�W=�x�=�̘=@V�;�)����[=l��<,G�<��<ҏ��,�=�|�:�/�����N�V=��7�P�`��f���m�w�>��r< gϻkO��W�b ^���ϻ �:��,=85޼���9������{L���o�Bi=J�轠01�t|��T�����Y=Ie�=��<��?=���=@R��/Ѥ��xG=�)�����zνM�$=_��=H�t�b���  ?��=H%�<�%�< ^Y<�D�=��U>������hB>���r�< �W���`���=�r�@�ļ��R:�$H�|&#�P�>&��8NU�>$m��L� -!� � � �v��c���#z�4�<�ѽ��N�A� ���@��<v�<���]�x�Ἠ �=P���<=}½��<s]i�%W��!��S�'������=c��=�c<�A="��=D�'=�.=T=���^�5�Խ�a�<�!�`�����*1���=���:�~� �;V�>��>HQ�����=��:�6��=�?�Q��q��Ѣ�90��(���{g;Q��=��k����<,�ؼP=@h0;̪��@� �����T��؊��N���F���R�|W�h��<�I�8�G�<������O��=���<넇=98�=��ٺ���<�s�����<���O ����<���(%<h>��07�<�&=��e<p���&��c=����v��v3:`���u�<`�<4��<|�)�T�"�p��; N!;�I����� R
;X� �5����0=��<a��� ��PHY����LT2=X����P=�O�=П
�~C<�� �0��<�r�R�9=L��<��q�@���h�<��=�9_�=��=��	��Y6���<�uc=�<|:�� �S9`�k���A<��6��>�<z���R����; �U<8�*��;�;=�꣼��u<k�=Ȏk<� z;�� h����<�w�����< �	���y��������bp�� ��S�=@����$��o�� �úb2=sh�=��W=\ߍ�����=``�;=�=��<�34��C�=�A=���K������ٮ<B-=ʒ���9:j�>�����;Â��pp��������<�ب��L7=и-��y�=`;[���d�X�DZ¼��v;���&G ���	���G��	=71o=�a�=0�d<KP�=�u<T��o�[=Qk�T�G�>�v�x?)�O�=0D>�D�C�ڪ=ڂ<dǈ<���<Fi='/<ˈ2>��̽`�-�W5�=X�<;�=��ۼH:� @�'�\S���)�= Eu��v}�83�=�����"�$���w�ν�� ��9��;�O=��q��D=x����"���O)�ī߼�����5{���:��<�V�=�K=`l <T�<i)ҽ ���_���	����N��\����=�����v����� ���0I�<`.�= ��`<^;��	��*?<ذ3<���xT���TA<��~=pRu��1��4R��>:&	>X�6��|>X��<G@=����D���Mὀ�5:]ν�`��q=a�= �M:0���dz5�P��; �v���|�\��������4���f.��0��T�������<��8��������<�
 ��i��D��<�)=�<�: =(�9��
�<��,<�h;�W� B�;p��*��`�;�6��`T�D]�<�}h��}߼�.F���:TЌ��UQ�8~�hv�<�^N���d;��<PG���<y���'���Z�P{�������0��D!��x�7<� <�|��L����Λ���
��>�<�∼DR�< �~<4U#�Rk�`F;��o��˼Pe�;��<X~9�8{Ӽ��;Bf�H��<���<��h��D��,��	V=�0+;�i>� �.;��X� �q��)Ҽ�����P�x���T ���;��z����< �
;4����˼��==H[A�04W��1�� 0;p�y<��[��93<0�)���PaB�l�ϼ����9;��ҩ=l%�� ��;Z�V�8�d�0~I����<h��<d��<�N�Xq���i�̛�pJ=Ȫg<��3�9l=���<�
޼�U�<@l; ��9�\E�Z���h��<���\�<����=�?����I��
�=C��2�X�V�z��=���03=`���M?�=?+;���
;�m�;�u����>�p��<\��<�2�<���Á
>�����W���/	=>��$s�~C0��Q=���<W�^=P���^=5�s=i��=����}>@d��Ã��؈�|��T��<�w�X�B<�� =� > �g;�v�����;�t���^�=h>v�Ҿ�w�=���=Y
��ג;=�O�@ϻi=�
�<&J�@� <���<_��z��=z�-�8�y=pn�<(��=8�v=eba=K.i=$a�=�`:=/W����=T���xG��þ�7�ӽk˂=�x��n4�]�-=I��Z���0Ľ���=�vL=4E̽��_=�<�5�����r�@6��p'=C��="<�X,��
�<���= <���<�[򼔀E=%q��L��5�<=p��0=��ͽ����:��=�u�=� ����<{=��?=�=����< ��9T���b�=�?���	=@"N;@�Ѻ�����=�1�<�d<hi�����<0+<d潼�O�<�h.� �V�D�<�:��=<$�ڼ�)���W2��h�<���;��;�`�<`�;�hb<H�?< �K��=��<@M;\��<���;��'=B�= `V����;�Q<�6�<��<�Au=��[<��<��<Ԑ�<
=d �<�V<����ȳ:�����d<]w.= v�<�*=@�滰\�;�Y;X�<F!=�@��<�8=�<�� =��n=x<lb]=Дȼ�C<��<�?�<Ȉ=`o��ά�$����=�N'�@�x� A��l�< ��<|��<��:`�:<�pg�p��<\�üJ�<@U���=p;Ȼ��<�p<P��<��A<`.=0�Z<��8<D%�<�k8=8�9<0?ݻ�(���<p,<ȝV� �;�JT<�?=�wܻ���=x��<I/��.� �M<O�<�O< �;
/�=���<���'�<KE]=�e����һJ콽��t�=�2=K�;�:=��޽r.�ȼ�<6�j�@g<h�K����=�'���H���;��-=�1����,��v<l㥽�#;n�V=�ʰ=@��:ͦ���">:�)�S���<\��Ԅ��2��+{�=���=�А= N����<&e�=� g=��`�₼=po�<�Uz<�A��g��O$8=dMĽpu<��;JV�=�K=�"��Ё��L�r��Q�=�~��a�<��"=���=ͷ��@h��#rʽ(�{� C���=�~�� ə�Xz׽�����"�<�~Ǽ�t=!�7=
��=�d���3<��=R��=A�<�>ܽ(�=>����oͽ�=ӽ#�� "����;��<,��=@.n�X;<�����3=d��<C�����\��t㗽;���(��<�J=gv�=�������T=]�=|�ѼTF�<����g�V=�2��D��d�̼0��;Y� =Ǿ�
�� ��=�IU=й�;��y<���<��=�����ջ .:��޼F�= ���"=�Ȼ J,:�Jt�8!�<���<H]^< h:=<�ò<ޣ���<x��<�D�<ț�<0��@��;��
�p��(�&<Б�;0.�X	�H��<� *< �7���K; Ռ����<�4�<�d�:�'�<��';���<ԛ�<x <�Ѻ�XMl<��_<��<�6=�~�;D��<�Q�< *o;�DJ=n�	=�<���h	���� �b���=�L�<�3J=��; ��9`�ʻx�<�JA=��7�?�b=�{�<cP�= &/9s+=�Ҿ�\�;���<���<V3= ���P���QҼdq�=���U�`�T��R�;XE�<�T�<�d1< '0<@y�;0
�<p 9�$��< =k= T�;���<�k�<���<�R�;f0=X$@<`�;�/<K�(=�;�黀�:�,=� ;�2�0�N<4��<���<�)�<W�w=6C=6�����;L�<m�=�DJ< 啻�0�=py�<x�0<�;�;-��=��`���D���*Y��mz=Atd=��4=,˛<�ѽ�Qм�Օ<�;��J=�D�;�*<0!�����-=���n=L����l[�NV���e�<��`=���=������� �>�@����A>=��0�$�t��G׽	t6=�ʙ=X�=�#� aP;��e=`��<`P��F�u=�O=��.=�S(=���D�=D���TS�<�m�Y'�=��=zd��6G���m�,��=�D���3�=,��<��L=��@O�f 8�8R���)�2�~��޼0��;tW����s� ;�����Bf=͟K=��=Z w��Y�-��=d��=u� =�����q�=��罿������W���탽 3=-s�=#&�=B�=i��=T#���?;`>Y��$Ž�R��J��`�����$ޚ�V97=/�5=��a=f� �뤮�4z=��
>xLL�+�=�x���k�=�E����ʼ| C���y=��^=����wG�t��=,��<�c�<x@�<�94=:�=�.��19��̀�~u:�lݳ<�H� $s;f�Z�����X� g�9��=@�E;�8H�z�u=����3&= �&=q�-=�;<�5��S�<�ڼ��F��V�<Dʹ��;�(ƫ��E=L��<�//<�ٱ;�"��R�@=@��:ļ>I?=P� �$k=�2�<X؇<��W� �켘�<��=dÚ< J[;p�D<Hx<_;����=.�Z=(�o���q��"]�N����Q���=��<p��=`W�<�f��n�A��bL<��=��<`���`\=�x�<g6�=��8�޼�=`U����G��Qe<�2
=�ӆ= Iֹ���|���/C�= T; ��PK�;(���v1=T8�<�=�q��@	b��
H= �2|S=��n=�:/=<��<x�<�8�<P<�<�E���_�< �F<8�><h���X!�<��%��5~��M{<!F=|E�����R<��"=xj�<�=Hg�=�5=����>=�V�<,�h= �h<���LB�=h�_�@��<L%ڼ�=���� [;����ʡ��=� =�6�<���Z|y��a���6�<��<��==lEǼ�~G������Q���%/=�Á��Z�=���X_�<�.���<Wyt=&�>00�ϩ=nm�=޴Y����6�=�#�����=!��Z�<1��=ĕ�<h",�$�����(=@�7<}�<��-=l�=��=(��<(�м��>��Z��c\<��4�X����=��J��_<�ǎ� �<�ث<�(*>n~<� C<�����[�� ���������N�u�p�h<0ݺ;ν0c`�JCG��廊�=�C=�X�i[̽t`�R��=�-=�Z=�#ɽh^�=��!��6p���I�����K�ӽE�c=��=z6�=��U=��=# =��;t��.D��|�ʚ�Ha+�4N�<�� ��o= �a��|���"k<q��=�
>�$5��=0&W<�s�=�?���`�.����D=.�<@� KA��S>,-��Xr=t�<��C=��<\i��' �� gٹ����h漘��*��ν`Z��-U�Z��`��U[=0�.�v�{����=��r��=�ۤ=��K=�K�;m�����<���T�`��+5=��x�`p;<����Q��=��M=�!�<@�?��\�kP=dpܼ�O� �/=�-���@$=�L�<G�=����Hm��< u�<�ܼ�'�� �9 d��C���g�=q�T=vx.�M��4m��4=�����L�<(z�<ҍ>h8�<#�%��$s�<��=��=Ю����p= u�<�+=�ͽ�d�=�� ���-���绘�I=K�=x��<ۛ��p�;nT�=�7�<Ȳo<զ<�m(����<h-�<��= ����8��n�=�c�7�=� �=e�B=��=\�����<@]<&o����i<�`�<�1�<2	�� ��|0��~S �y�^=#�=����>� b���,=,��<�B >��=�\�<HD���g=X��<�\�=���< 
����=p���`$��cH���=����8W�< ?�� /����>0�m������a�,a漦{`�&=P�S�&=��T�R�;�����"B8���_��^��Sf=i������пq�P�.��e=1>��ټ��O=���=�b���)�X��<��Խ�j�<b� ��7�7�=D�м`c��8�V��//<�@<�X=c�I=t-�=�>P�A��ٻx">�O�;�'�:�H��h���9R=�����A<�/����� �-;�O>d�
�|�M�S}�\{��ˢ;��� q�g��~׺`&��)���.R��Ļ�T~켼�H=�`��n�gв��>�*�=(x�<l��=�ƥ�FC=\F���x�c�T7	�������=���=��H=���<~ذ=4�={�?=v��h����L���t�e����peU��s�����=�}E�ӿ��j=X��=r�=(�����=	�<)>��ѽ��� ���<û��<����L��<�N>@�-�̄�<�P,����<�]h�.Y���.d�@�f��c:��W���'���"���ώ��&�P$u<6�L��� ��,=0ݻ���^M�=nS���V=��=�_=��#<JC�0F@<l[׼̝Ѽ���<�h��2�<����~�+=|R=�?,<��l��q�̒�<��!��o
�H�< �>����;�?;�=L嚼�鼘�3<x�8<~x��*��J�@����Ɂ=�N�<�.?�uԆ�X��Pth��dڻ�E>��1�;~��=��߻B�=��;V��_<+K =��=0�;P'<��ɻx�t<Qͪ���== �<(F4�������<K͏=��<���� V�9!#=�r< <pXu<�]W�8[��du�|h�< A�;Ќ�;��=�5�@-�<&��=�Z�<��;����8A�<0D�;�׎� j);�F�<�9W;\��２ԃ�b��ҭ=���<���^.J��ʚ;��<t�<1��=j��= 0a��.���=��;��=��<l`���>�=hC0<
���3�X��䔀<i?=fØ� fE��>�wW�WD��P$��p���]��Q�d=��%�^Gn=(���(:=�,�`�(��IR��#�0塼p#u�zo�hY9�
����=��l=x�,=��P<��h=b�����X�<�/����g������<��!4=p:�����+�<�i�����:g�=�xX= ��;��=\4��|aƼ��Y=`�;(� =t�l�?���̼V�d�lj@�I�=<Žh�\��cX=f�+0�	����8���<T����[<�G�<��`��S<��f�,�+<��1̂�xP-��@�� ]=(UJ�c=%=hU=�1X=�ý�I<:C潰5l�����	��K`= ����.�TKռ@0�Pu�;B��=�`���k=����b�L�̼������ؼ ?�0�W=��	���0UO<�/�=��=��Ӽj��=�=C�=KB��-��0۽����ͽ�g�a]=��B=�/�<(�(�\?�p������,������K��좣�hؐ�������;@@!��Y �$�<��>�(GI�P�<��;MD���X0�@�:@n��yj_=\q�< b�;�<�#k�Nb�P��;�@��Q��� y����� 2M��<eļ�g�P�b<�W� |м�+<мP6<�`$~��=h`� ��<@�#�x^-��e�:�"Ѻ �ӻ��H;�KB�P�;�rP;�_������5�<���; )S�`$O�p[�0Ҋ� dK��*���*;�m����;O�;�Zu<��	�d����<z���a����:�(���x�l���S���,=pC��|�����D�T���`%�$⑼����B�ؒ���S�hV<(%
<�-=8����h��,)ټ���=�e�;��/��%��-<x�1���5��!g�p��;\����s�����0���"���X=�<�< ����0�\t�<@絻�dA;��<|��<�n��8�N�� o���y�H=�<0qL��=�p�<.�=��4�t��< ����<����`I[�p<p<����J�&��=���{�lR�=wf����-���߼�=(P��Q=�?P<
��=����Ѓ������C�<t� .�<��=�s�4�`�=�=�*<�9��E6=��L� /�"�)��\�<�ʥ<���=Pe���#=�m>=���=Ҙ���
> 
<Ro�ړ����a���;.M���Q<y�%=�k=�#�<2VD� ׫���)�AC^=L����X��հ<�}�=Ͻ�S~<�н�e�,1.=\�<J>����;6u��PG��y8<��+�x=�<iK�=���;�xn=U=xE�=�]�=a8����S=,�潼Oɼ\���/惽�>�<�0���ۼc�=>B����Z�̞a=��K=$�p��l<���܎����jP��(�<�
0=���=�B���B��@+�;@��=��;�0
=��6<�= �9�ļ?��=�!�洂=R뫽��7�s� >zڇ=`U><�)�<W�=P�:=��� �90to<����tT�<ˀ���2=j��G��\ �x7<�z�<(o�<�r��X!<�M<J�f����<�`л�<��6�X�$��!�;n���G�#��`�$;Ќ�< �ʻH��<p��;��X<�x�<������<� =�,����=��)�Q�<��=�W���|+<�:�<��: $<�;k=��;`�缈0�<��;�/k=$��<�d���5v����x�ּ�mk�:+= Թ��X=�~!�𰩼u���$��}�f=����7=�K�< �/=,1�(�<�Q�@�����5=�&�;�G�=��~�,Dʼ( @<� �= ݋��̝� �X�tʇ�`H=` ��8� <�}<�犼x�[<T39���<�5�;l
�<P���
�<�4#<���;h�$��`�<�P�<XE�<��<b_6=�Z�:XGs��Xw��cC=���;�<!�8v�WQ<��c<�I<�=V�=��pL���λ���=X}�<��\;Zs�=�<�� ���5;�C�=�F��Q=�y������t	�< ������<���=�����P׉=�Ѱ�\"��t�!���;04
����A�<��<��^��ｲe-���q��.���=fj">��^��s���>Ƣ3�v�|�R�=���- �'�ڽ���<�u8=h�=�*���q<Z`=��O=�2����
>^��=����"����
S�0��;��＜�<Hw?<��=y�H=W���@�E;|*ǽ�Od=x+����$=`�<�i=�P	��L�I�ֽRD{���<�&���zp��D`�u��py�@I����O�= �Y< ��=ʘa����<
�=$1�=� �=J0�$z�=b�*�J�il���9Ž�a ���;��}5=Z��=���5W<�Y���=Lk=����a���k���Hɽ�ŽX����<�D=�Ĩ=5L��U*��P��<\�=� :�
=�=,Y�=�����7�e�&=o�!=&��=2��g��.>C�-=<��<�c�<l��<�t�<�0��<��hx�<�A�di�< ��;�#i=�6� �»x�-<`\���u^<3&=�ر�`m;�N�<�$��H�<x|<(^=�
#� <�� ˺��@뙺p7�� ����O<�>��s�<ptC<�-���<��<O�<TE�<`{W�&=�)9����;�%�<��<��i; 6�< ���d��<D�<XK�|7���x�@Q,�Y�=�f<=�&5�ပ������D��g���6=���<U%=���<��@L�,�����=�7$<�%����=0��;$FZ=:/���:��.�x�$<,�A=�/:i��= C�������<%�= �l9�H� �i���0�t�=HV�Ȋ�< �u<�Nd<ڤ<��p��<*�8=$��<���h$�< ��<|��<�k���gc;ޥ,=�S<�:�;q=@�;0q��P�;J��=�<>!
��#�:���< �4;�� =�ɕ=Y+=oD�09�;�:���=X+|< !d�Ҩ�=0�?<� A��.�`p>P��5=�5��t��<0�><�eZ=�M�=0������B�=�"p�@;��)�����;�J{�ޮ�=0;�<�<����q��er���,;���<V�R>>e��Ȍ�fb�=�O���P]����<lJ��'��B��@�Z<?F=1]�=�!��`�W��HE=|��<pu<5/>�X�=��{:ĸt��� q�<p�<-d==`<Ļ��;�"�=��;� �;���g�t=�A<��= ���,=�s��?߽ާa���s�X���z��0��;0C��(a� RL� )ȺTP����=p�-�r58=I����<���=�n�=��=����,�=�tȽ��_����`���J��Ќ�=Ɯ=��<�Sf=�J\��z ����<�ힽ,���R�$����Bg�df���<��>=CG�=�ý��޼>� =�B>bn���d(=G�h=h��=d���*:N�X|<���=��=�ϵ��7F��RA> sT:r�-=�,�<�@=�\
=���������<Z|k�0��;@�,�_�<m�Ͻ䝈����L��H�z�"Ί=p/C��e����=;�,q<=�<9s�=<�ʼ ��L��q�4�����k<�u�����<�O���|=-�
=��J<���;�����I= j<�2�ZP=@s����\<�&=8�=�"�:��Լ�cݻL2�<�\�:�����d����ɼ��I�2��=om�=��r� ����PEM�FU���n=���<�j�=U�D=> Y���?�xf�t��=�=xt�l��=�\�<�+l=[�̽�<(0J� 	�59.=�u�<�u�=th��
I��4=�>T�<p<�FS�C�����R=X�N�@U=@�<��j�;=����\c=��=�EF=�ޓ<�V<�d)=�D<�U� (��V#r=�ۓ<0f׼���<0�J�tS���+=���=;��.=e��[��SS=��ZZ�=�[�=�\=�j%�5\=0<�;�	>��<<��� ��=8h���<�N��x^'>$�ֽ�7=���b��ϥ=`'�;M�=@σ=�ͽIr���(�=B��0����/�
�`������=�=�R��̇p==-(�Z�4�X"�<P�<b�=�=F>�A�� C�<���=�b���D��@k<\�4���a<h��������	=6�\=t�\�ؼXU=�EN<4=���=H�>�e�<�F����;�Y�=�cR=�A=�𭼸�O��җ=m`�1�+=t�ֽ�bY<$�Q=+!>2B=� �Y;���u� ��^�=� 8�&��6�7= �J9ڊ�d(�< K�����o��=������<����bм1Э=yh�=�*�=_w����=����:�{�wT��_9�0��<��=Y�=�I+=��=���<��U����<��wG��uǽ����a�)=_W<�J��"rV=1nW=����Cl=qV=�s�=�l�@�^=pM�=.�>��������쮼�L=%^�=����pu��	=>���Vh=(��<_v=df�<�<���ﰽ^�<�5���g��̼��ٺ����ݼ R��.�м0�HO�=H�μ-O��5z�=D�!>�=XÅ=wt�=�;μ�V3���<�W��XyW���=6�V�!T=�+��!�=�,v= X�<�K-���.��Jb=���:�/u�ji9=ZĽ�<� m=��V=��#;P�L�� �;���<dμ�����.;
�+0��8�>R�=6�{��|����镽�`� K=��< ��=�>C=)���'c��P��;B�=^�=(���f�=(��<�k>=���M�1=@��,㦼���<[x=Tu>���>۽(Gm=�Q>�=�6�<�_<�^����=8�	�Tlp= �?<����=����r�=�|�=��v=x��<����!P=�,B<������	�Ќ=d~�<����	�F����c�7�=�\�=����)��TԳ�Wyq=@��->�_>�== |�:6ȁ=��;��>�J<����L<�=�AӼp��;<�����=�6��N;=��A�@q8��/`=��׼8�O��Ҭ<��@�!���~�=� ���l������V�\��� Ӭ�xZ�<�7��Z	=����6'���=�@!;.�A=��!>ܷ���K
=j�_=����qO� 2N��w���]u=���h���=�<�z���� _�; �"<7(N=���=��	>Nr==������<d��=�/h= $�<��@��!�<Ȱ<齽�V�=ڭ�����2�={�=��۽����h:�0P$��g<�9��W	=�}~����<��b��`V�<�!���^�BG�=x���=՞���7�v�W=|M=�C�=n�&�g:�=�n
��&=��?z��5Ͻ�"�=�[�=�=��<�c=a�<T��<��}�p\�<�sɽg����妽(��<��=^7ɽ�4i= (;�+�xg�=r��=�k�=,C���=Ԣp=�>&�N����������ؐ�<h_��&	
=OX">Є��F,=�1�</=���2�ٽOV�� s�;�du���,�T���x��Y�T�7���|<�7w�@V����=�_y��Ƚϝ�=�D���b�=�=���=�j�����<�l��5;��=�)r�he�=H����=J(�=@r <�P�����z�=<z��~,���< ��`�X;�I=#�S= g�;�'#����;<�<�O�<"��6��4�m땽f�>�o�=cc��p��h���j�*f����<�{�;Vy�=8�o<�菽r�v���x:���=���=Ц	<��i= G*;\�<lE��m�<nƼ����b<�O�<��>�n<\�ѽG�1=ߐ�=��=��<(9.<��0��;x��3�+=�}�<  F���Z=X�:�W=+�=D>=��:�P��E�X=���:҅ݽ0�/�dO�=���<����ؤ��>���1?����=� �=$�C��ѐ�t7��ʀ)=P0ǻ��=�1�=��c<��<hY�=Pi�l<>(lG<��ʼ;b>= ��:�OV��r׻���;��߻Q>l=���ս~\9=�	Z�ΑE�h4u<�����"�=�-f����<0=��OO<@$c�|�=�u�P�O�2�Dz���h��#�<���g=#�= �� L�;�/�<~�|��3(�`>G�DKW�~6=��½�}����;`gB��D���;������V'�0<=iP�=�^7=@k<��޽�$X<��<��<�= 8>��Y���"�E���,��< ,�;���@p�m�<=D�\l����н�L�`'�;(ɼ�'3=���< ����k<𩄽|{�<��������t���Q<��p�=
e����;`&;At�=�u�="�h��:$=~ ����=�9��Ԏ�LqǼpi�<x��<@�I�8(����� �j��\{=�۠��Ŧ=������ܨ��1��8s�<�F��>�=��f�1���=���=
%@=�ϼDx=�P==P�=�	���Ͻ
H���n���/?��σ��r~=���=P��< ��<�̐<x}><@���-��bL���;��C��q��T<K�T��<a1���G9���<KZ��x��N�w=��<�F����<b�^�\
=�F�=`�x=����>
��%�;.�h�H!B<�=��V���=D��� �+=�c=@����(��u��Eּ�&�������0P��7��(0q<0�'=�R<�!&�@1ʺ�<�C�����о��O���\C�D��=��<=����ҽ���<@뾻��P?v<�j�;�(=0@���F���u��>�ʴ=�Đ=� ~<p+�<,�pb�<Q�˽��@I%��'ϼ̺�< ��.��= ���$Dj�h�0<{� =��<�gV�`.����  �4TL��x�<Ԑ�<�h�<�Y�<�d��gN<���=�¡<H㨼w¼iR= ��bx���ہ�Uu=�_�<ZՌ��]�;x"�&�z���=K�=8R6���f� ;�:t �<�H�-�=5S�=�j��@��:=p1���B�=��<LȆ��B�=��=��7� �M�j�=BK��w�;��k� y����¼�h����=q]�� ݜ�s�g=�G[�:o�8��(;%<��a��=��4=HD=(e\��q��^��cV=� �@�:�=𔓼а�<�p%=d �<�[ӽ��*=l�R���<DX��t#�<`#�<�6!<���<`��;��<�=�y?�m#�=>in=�o�������;��K�<(��<�̴<��<�7=`~�<4���8j��+u� J4=|���ذ����<V�=�M������W��{��cl=H�r<�4�6'=����V
= ��;@揼���<�<�=�xļ��,=P��;��#=�m�=]ӹ��
�<����l�; ���H 0��<:9�������c=8r� N�; S2�(V�<K�2=��~�=r�Q���F��, �������;�v<�=�6��X�<�xd<B[�=8<���=��Y<���=h9< 6��W�=@'��x�B=<���+Ǽ���=H@�<���<|:�<��S=��=�c�����E�<����<��м��g=&(q���(~5<0��; {�96gJ=j��Ht����<�K���(=��;�= Dh�0���Z<<�6$��f�;�N���T�s3=�۔�ۅ=l\�<hY�<4��<xN>��6=8
=��ɼd�?=����+=V=�r"��3�< �:R<h�&<>
Y=@V���(���<�g�����=X@,=h�ӼG��P��`�L����JB=�(<��^=@m��t�Ԉ�������~�= ��Ԓ�*=I=���<0�$=�@��0p�< �r��;��L�B=�q<��=�N<����m�<ڢ�=�6ѻ�΃� $�;D{꼄{=��-�D�<���<�8P���=��U��r=�C�<��A=0.���v�<<�<0�<H߼ଂ<eCH=d��<�\<^�= ���x�t�P�.<�"=0	�;��A�K��d��<�VK<���<�ܾ=-="� � ��8!��ا�=4Y�<�j�;O
�=�g=h���2�f��=�z�&�=��`ѽ�m�X�ü�(w<D�=��b�w��/�=���=����)� �)��x�p^[<��2=�˔<H��ԓ�rh1���q=2J>�`}����=@�g��<L|�=��.
=�p L<
����M=m���җ<z=��<�Z��X�'�L��<d7�<�!��҂�=a-�=�n׼D�𽀙9����;tQ=`��<�<�������<RJ����;>���=��;�P�<��<�j=/��{k��_��2�>�_>= GB�$�� ��:
�����=�7��l�i]E=@� ���B=~�l�=8�d=B$�=���=�oڽ�b=T���I���4�v�!�����:�M�<��<r��=`��;�aK<�E���}�;e�H=����uýַ��8�%��jҼ��z��̿<�6=��W��9<D��<��=��ݼZ{{=�#=W��= <[;�&�S"y=���<�e�=<E��,��B>���<��<�<�	=�=P�O��c���'=����P8<���:>�=VZC��4���=�R�;���;�vb=`CO�����w
=�t'��P=��<ȎV=�	�~�=�;r�����<�~:��g;�E=� �|�<@�<���;pKK<H��<���<�=�Q���=@w-<�w<��==���<�<T��<Pz�;@6$<���<P�r�paؼ呼P�;b��=^�v=@⇼�3��Ė�< (h8�����\4=���<�
=�0�<J�%��߀�Lz��j:�=4�=PU�����<���;��P=A��K����@ƽ<9j=�~�D��= #<���<�!d=��=��G< �� �:�~i���<�߼ =m�=d)�<0&=0�⼅O=#�s=i�=�?u��?Q<�.2=�=`�Ѽ �k���=�B< �;9> =��<<�\<	/=�Ǩ=lW�<�X�����< RK���>=y|�=~�= ��:�T<8^��0�>���<� �;���=tĹ<t��������>���X=<N��B�8z(<�$���<��=uF��QGǽ&�=�����{�HԼ0�ֻ�D^�0l���=p;�� @��<�r�>���p=4�޼ b��.�!>vn���U<�Q�=�H��=���X��*�U-=Oi��;�<\=��<0"ֻ�iX��5=���<@$�<&9>�>0�3�V^ս�]b<���;9�n=Kq=��;;��(�ы,=��$�<L6ڽ )=v�<f=�=Pz<��W=�-��'νֿ�$U��= ^^�@&P<ЅA������pu=hP{<Pj��@]=����Y =������=�`�=a�=�>WMĽe*�=^��� ��9WC���~��O�X�m���O=5�}= '�<���<��6��ڰ��4c=��5���p��볽�4����X¼�{*=�m�=뺠���=hZ=/I�=غ ���j=�=l=�7�=@��:����A=D�0=A��=<�/���¼ˈ%>���(��<�<n�=˗=��'��S�� =����;���;z��=.�v�����j�< �ۺ�<�:��g=`Ԅ���꼚<=�3t��,�<�!�<�N�=$���^`,=�ρ;�v�<��<�		�@T��\Cf=����B�=`H�<��~; 3<�d�<�S+=�� =�A��R� =@��:��;��Q=.�=\��<���<  �:�n<H<ϑ�6ּ���`~ܻ�4�=Π�=��u�͜ǽ��< T�8�����I=`[�<
� =t�<2�8����`�����=k�=���d�= �
;�	\=�I���D��di��|�<�O=H�Z�tv�=H�%�� <�C�=瓏=��< &W;�,U�q���<���Cs,=�j=�D�<���<@���=5��=o}=`6��P�c<�C=�d�<�  �@�����=p�;��!�T��<��l�x�<_bl=���=�?�;b�0��Q�[O(=����+Up=���=�}.=`��;�·<輲��> y:<�7�;ܳ= d�;�<@ �:��
>V�����< ��:�a�dL�<@�;���<f��=����鼽8�=CՌ���n��缨���,B��(��0I�=��ּ ��<$����ټ�u= ��H�)��� >`ϋ���=��l=~h]���#��Fݼˏ�B�b=i�����<p҅<�=��`_��J=�Č<�W"=`%>�M3>x�<2	��ò*=��=_��=���=���:Xb���^=�����;= �Žl��<�=	w�=6+�`�d<���-�� F�;ʈ�ukD=hu:���#=X\"�4M���=ɋ=(!	���p=8���0b=2^��q�<q�{=�׊=�L
>�)g���= ���Y�<�����¼7���\��< ��=�`�=�,:=T1=�;��2�w=X��<c���ɹ�NG����=��;�r-��EX=�=�P��
@�=\q== �c���^=�k�=:�=@��:J���<��=��=д� 0���V>�.��HQ"<�J�H�=
@=�i���+��`Z�< 3^��K*��S;J�u=�i��sٻg�<��<@��;~�_=�Ă�����Q�<&S$� J_<H��<�xz=�\���z=p�B<V��U�+=�Њ�`	(<��U=�t��@��;a�< DĹ �p;I�=z^=TG�<@ݎ���<��;А���f=��=���<��<�K;8�:<�W�<x7
�<͘��!�Ѝ#<�_=A#�=p��E���%�<X1]<\��b1=`_�<�?�< <�;B����v;@��x��=���<0P6�h�<��E;��F=��Q��*��������<�((=�����= 7���Ts<Bj�=.1?=d��<�!< ���^� �B<���.A=��<�=��(<P�㼬��<,�v=�R=XP���
�<Ѵ4=��<83ݼ�Y�+}�= :g9p=�;��<Ќ�; �7�`��=䎹=H�<Z�������4=D�Լ�=�aa=ڍ(=8Q1<��<H��d� >�� ��;���=��!;���<�[g<�{�=h:����<�=~�&���< *�pk�;�'�=v�F�c]����>�V���h�,'p������M��R�<�=���اM<��ؽᨼ7m=��0:���<��>�S^�&G&=�L=
-^���#�������>%�=9�׽ ɲ:��t;��<0=������|k�<p�<��V=M >^0%>�h�<Rͽ�i=�sh=��=�4=@z��,�ɽ�=�<W�(��=�D�,����y�<�2�=_����y����[��`9<`��"�=�;��f<�3�;�սD�=�"�<�|�(>�=���*_=~�Y��K;�}9=4y=��>��w�n��=,��A�5=�|����
�J�W����<p�{=坍=���<���< |�8`��;#F=^CG=ȳV����d��<�C�<h��wI=�k2=;�ͽW��=���=<�= 6];I�=jR�=���=���i<����@s��穐=�����<�><�� ����@ӷ<�yl<D��̝���;�!�������Լ��<=�dG�����d��< R�������>=��
��X�x�B<��ϼ�;�<"
=��Y= �۹�C=�:�<6�"�/88=*𻐝��Ȑn=����O<� =�$�|���h�<��<P�9< @-:�ͻX��8�E�=c=,��<d��<���<��z:��V� .<���Lۼ���� �:��s=K�u=�D���M��l�<�<�<$�޼���<��'< `�<��H�"\+�`VQ< �D����=V_5=@l��̓<p�C�{�=^	q�$Ա��HռH<�G�<��0�o��= �� ՟9��h=��<d3�<�x; ��vf�Н�;�C�1�<���<���< �|����h6<�`X=���<(7���D�;�,=x-�<��3��[��d�=��;�Z_��	�<��u�k��u�=���=H�5���� aI�]A=&���	=��k=`h�<�6<`< ]/����=��";L��Q'=���< pu<��< �<�����0=L�<@>ڽx��@�ͼh+Z���O=�\��7��%M>�P�������7��p�<(�3<��M= ��|@�PV/�Dt6��O(�� ==��7����<��=%�H�<h;�<�+H�jp�����}=�䧽�D����Lޓ<���豛�@�0�P}�;��#=g��=O٣=L����%޽M&O=��;\'=9== 4�^���8���n����=��o;h�A�8h���}<��e� �g�mӽǿ����Q;�<<*�=�R�;�Z�;��<L������=h��<L��hoZ<ԝ)�ۼ�=Z1�$<�<��5�ی=�I�=�t��~X�=¼��%=����s�`��� �<��)<��<�; ͼ�\����<�0=���= `E�6Y	�X���,�Լ��S<�ƛ��I�<V7=q½Ƹ�=���=?�9=�l���C=+_=���=�M>�z�l�H�,��c�P�<��/�!� =�KX=�9�< ���8?<@h�@���XY+��{?��,b<d��4�d���e>=�\�\#�S�<�*�0��x:=؈C<N�F�0����?¼��:|�
=S�Q=�鳻pf�<��D<b�?�pO/=p��;�3����.=`n��,<�=�`��<�� ��<Эٻ��ٻ��!<<�ļ ����Nڼr=ȁ�<lC�<,�<p4������@6;hxR�\Z���i���:ʅ=T�1=	��P���m(=J�-=`�Ӽ�¤<�|��@�O;DH��6l����<d���h˛=��Z=@�; ��;X[��j�<��d��W� ��@�c����<�&�b��= �� �<"�= 8�0<����f���y� ��;Jlb���;�&=���<@M���iۼ�E���ZX=p~�;�Z �P|ڻBJ=��	;@�B��aC�0�k=�Z<�s�!n= �>;��#�hrY=���=��˼2N����L��<���`M�<�B=PI�;H�<`�;|G��w�=���;���*
dtype0*'
_output_shapes
:�
�
siamese_3/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
data_formatNHWC*
strides
*
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
3siamese_3/scala1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_3/scala1/moments/varianceMean*siamese_3/scala1/moments/SquaredDifference3siamese_3/scala1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala1/moments/SqueezeSqueezesiamese_3/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_3/scala1/moments/Squeeze_1Squeeze!siamese_3/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
�
&siamese_3/scala1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9
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
ksiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_3/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_3/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Nsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese_3/scala1/cond/switch_fIdentitysiamese_3/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala1/cond/Switch_1Switch siamese_3/scala1/moments/Squeezesiamese_3/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese_3/scala1/moments/Squeeze
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
N*
_output_shapes

:`: *
T0
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
 siamese_3/scala1/batchnorm/mul_2Mulsiamese_3/scala1/cond/Mergesiamese_3/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_3/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_3/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_3/scala1/batchnorm/add_1Add siamese_3/scala1/batchnorm/mul_1siamese_3/scala1/batchnorm/sub*
T0*&
_output_shapes
:{{`
p
siamese_3/scala1/ReluRelu siamese_3/scala1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
�
siamese_3/scala1/poll/MaxPoolMaxPoolsiamese_3/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:==`
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
siamese_3/scala2/splitSplit siamese_3/scala2/split/split_dimsiamese_3/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:==0:==0*
	num_split
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
siamese_3/scala2/split_1Split"siamese_3/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese_3/scala2/Conv2DConv2Dsiamese_3/scala2/splitsiamese_3/scala2/split_1*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*
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
^
siamese_3/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/concatConcatV2siamese_3/scala2/Conv2Dsiamese_3/scala2/Conv2D_1siamese_3/scala2/concat/axis*
N*'
_output_shapes
:99�*

Tidx0*
T0
�
siamese_3/scala2/AddAddsiamese_3/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese_3/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
3siamese_3/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala2/moments/varianceMean*siamese_3/scala2/moments/SquaredDifference3siamese_3/scala2/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_3/scala2/moments/SqueezeSqueezesiamese_3/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala2/moments/Squeeze_1Squeeze!siamese_3/scala2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_3/scala2/moments/Squeeze*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_3/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_3/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
 siamese_3/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_3/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_3/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Tsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese_3/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_3/scala2/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

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
siamese_3/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_3/scala2/cond/pred_id*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_3/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_3/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/MergeMergesiamese_3/scala2/cond/Switch_3 siamese_3/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala2/cond/Merge_1Mergesiamese_3/scala2/cond/Switch_4 siamese_3/scala2/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
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
siamese_3/scala3/Conv2DConv2Dsiamese_3/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
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
*siamese_3/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala3/Add%siamese_3/scala3/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_3/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala3/moments/varianceMean*siamese_3/scala3/moments/SquaredDifference3siamese_3/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_3/scala3/moments/SqueezeSqueezesiamese_3/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
ksiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_3/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
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
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_3/scala3/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_3/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_3/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_3/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
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
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese_3/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_3/scala3/cond/switch_tIdentitysiamese_3/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala3/cond/switch_fIdentitysiamese_3/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala3/cond/Switch_1Switch siamese_3/scala3/moments/Squeezesiamese_3/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_2Switch"siamese_3/scala3/moments/Squeeze_1siamese_3/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_3/scala3/moments/Squeeze_1
�
siamese_3/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_3/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_3/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/MergeMergesiamese_3/scala3/cond/Switch_3 siamese_3/scala3/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_3/scala3/cond/Merge_1Mergesiamese_3/scala3/cond/Switch_4 siamese_3/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
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
 siamese_3/scala3/batchnorm/mul_1Mulsiamese_3/scala3/Addsiamese_3/scala3/batchnorm/mul*
T0*'
_output_shapes
:�
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
 siamese_3/scala3/batchnorm/add_1Add siamese_3/scala3/batchnorm/mul_1siamese_3/scala3/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_3/scala3/ReluRelu siamese_3/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_3/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_3/scala4/split_1Split"siamese_3/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_3/scala4/Conv2DConv2Dsiamese_3/scala4/splitsiamese_3/scala4/split_1*
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
siamese_3/scala4/concatConcatV2siamese_3/scala4/Conv2Dsiamese_3/scala4/Conv2D_1siamese_3/scala4/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese_3/scala4/AddAddsiamese_3/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_3/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala4/moments/meanMeansiamese_3/scala4/Add/siamese_3/scala4/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_3/scala4/moments/StopGradientStopGradientsiamese_3/scala4/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_3/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala4/Add%siamese_3/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_3/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala4/moments/varianceMean*siamese_3/scala4/moments/SquaredDifference3siamese_3/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala4/moments/SqueezeSqueezesiamese_3/scala4/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_3/scala4/moments/Squeeze_1Squeeze!siamese_3/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_3/scala4/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_3/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_3/scala4/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
ksiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_3/scala4/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_3/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_3/scala4/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
usiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_3/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala4/cond/switch_tIdentitysiamese_3/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese_3/scala4/cond/Switch_2Switch"siamese_3/scala4/moments/Squeeze_1siamese_3/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
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
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala4/cond/Merge_1Mergesiamese_3/scala4/cond/Switch_4 siamese_3/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_3/scala4/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_3/scala4/batchnorm/addAddsiamese_3/scala4/cond/Merge_1 siamese_3/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_3/scala4/batchnorm/RsqrtRsqrtsiamese_3/scala4/batchnorm/add*
_output_shapes	
:�*
T0
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
 siamese_3/scala4/batchnorm/add_1Add siamese_3/scala4/batchnorm/mul_1siamese_3/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
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
 siamese_3/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/splitSplit siamese_3/scala5/split/split_dimsiamese_3/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_3/scala5/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_3/scala5/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_3/scala5/split_1Split"siamese_3/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_3/scala5/Conv2DConv2Dsiamese_3/scala5/splitsiamese_3/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_3/scala5/Conv2D_1Conv2Dsiamese_3/scala5/split:1siamese_3/scala5/split_1:1*'
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
siamese_3/scala5/AddAddsiamese_3/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
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
score_1/splitSplitscore_1/split/split_dimsiamese_3/scala5/Add*
T0*M
_output_shapes;
9:�:�:�*
	num_split
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
score_1/Conv2D_1Conv2Dscore_1/split:1Const*
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
score_1/Conv2D_2Conv2Dscore_1/split:2Const*&
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
score_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*
N*&
_output_shapes
:*

Tidx0*
T0
�
adjust_1/Conv2DConv2Dscore_1/concatadjust/weights/read*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0
�
Const_1Const*��
value��B���"���E-�,�׽�i���ܺ����;=����z��=��<�*<�J�<����ɢ�@��::�d�W䊽+ɽ��׽ ��;��>v���*m����`E�=� ɭ<`= f׻�j�O-���H޽��@��%��q=�,���;�>�8�3�0��*���M�òν��X�;��6=�'ۼ(���t�ͼY;<@���~.���ɽXy��g���آ<�/=�Z��ҭ�=b�!�{X����=,W�� �n;(3��귽�wO�����,ȕ<t���"�=U����0=�L�=A:��ڻD-�<�v����;<|aܼ��{����[9��O�=�fƽ�6P=��6=
2}=K�$=%�������=��
�{W��8/h�8�x�����g���x_��н΄��ˑ�� d;ွ��<آ���P@n�H�Ƽ����i*�}�=���Փ��������șw<H락'ד�_헽酟=�������؋@�Ļ�`A';���� <�8���==�9=�= �{<�d��hT����O�2���<d_*�
:	�^�f������{x��G/�����X}<�2�8E�?��=�8����=���<�o8���<�w���>=�M�< &~��:=��漖2= �;��|=r�=�>=`�<������=XF���.�No<��l�`�F=b�I=���;8C<o���ۿ<Ⱥ<�؛��)�� ����;B9V��-�=���<��|��<J>�|����cC=���;Q<m��=�P�<�׼cۼ�~�<a�(=��=��˻&9�=��<@j��x�>L�=@#�6<����!�=��t=4��<Tɽ`�Z��FR= F����2<�D|<@07;��<&=(� ��:��q�Cݏ=�0���o=���:te�<I�<��̼ ��8xYD<hOL���.<���;��<@�@�h��<��\����Ȫ�<ԩ������tmǼ�`���<02�<�_=���=@���@KU�DD=�Ţ;��S<4��<ȩs����.YĽ�s�;�/ȼ䐵�n�=|���c�=`C�|������<����0w�<�.���R�ʄY��)��h���[�=�ى��l���]�T��<T���԰�<�A=
����|����q�����`7��9۽ ���V52���<>:��$����t
;�[K���;�.���ɽ�'�@�^;&8�
�%���]�tb�м�PdZ� ��4�0�;L���W;��ƽE��=P���߽"��=��<h
��'������#s�������֯<�)��0M�;�f��}=`���8���� �89�NE���<�I�x�,�����s�����=rm�~�� �u� �A���0=Ib�0̽�G�<g�޽1���8����=����m��A���q��d˼�������Σ� ��;�k
�F���^����Y��k
��f������2N�~�(� �E;�B��`e�; ���=�Ѩ���ÿ�QSȽ��ӽ�A�Sk�p*�����z������<܍�<��+=0�j<H�U���;�B��߭�p�ƻ����@�l`��X�%��A<`G�<I�dB�<�,L�(A&�]�R=�M=$�u=O=o�x)=�'�1�=�F�:�}��+d:=����=,�<<��<ћ.=Ts�<`<�;�-w���= :/:�����;0�	���2=U>g=��T��G<8�%��M=���;���;��ںT���< ��(��<�<Z��`�8<�� �Ƞc�^�=�K;���<Ld= w3;4L�����h�=\��<���<0�t<��B=Ԝ�<('
����}V�=�;0�㼨2a�� N=^PA=PJ�<�tt�,���R<`���e˻��t<(`�< �b9���<�e���5<|[����|=��4��=p��;�7=�b�;��2����;<~�<�!�D��< (�; ϖ<����3�<L����䲼X,=`�A��Dں [X�����5�;r�%=�-�<�,�=��m���\��*�<plY<H�<�<@AI� �������66<P��m��я�=�ӽz`�=P��<0��;�=<�� �n�:�\)h=�I���V��HyM<p�ݻ춑��ͩ=t�E��2����>���>��⛽�-U=�|�<h���Й!��^��.l����<��d�ະ�h'����=_D��Ϙ�PR�����X�<��I�Ƚ��@��A�]�����t�XaȽ�q����<��j=0RA�O�0=����D<�콖jb=8C̽2B?���<��B=�׻�C��b*������ýt��=������<���<>]=>��k��T��<��=H	X�FI=W�0
����L�.�\�~�;�"꽕Ľ�D����[��i�<h���z��3��
��қ��Oʼ<�=$ּbA�� <О�|��<x��H�@�`�P� _�<�8F=�畽dg���$ҽ|D��x�R�H��b��.���"i=W����b�<0�U�&�� _�u^��QR��F�	���NU�l�"� Tκ���bR� \���S��S�0=\�<|��$��<bH�P�Ļ ��;xX̼t��< �O< T;(��<R�H=`�G<�2h;B�n�@��;�V= �=/JQ=���<��w�J= ~K�t��<�� � �Z:��<H)r<�!�<�u�<�$�|t�<ĩ�<@�};���³=���<@��;� ;�=��J=G�V=d����G<H�^�l�= �9;�=��,<|���ds�<�B<`&D��� <�si�-�=�2��oA���=��M;��,=d�<�l��@��(�<d�<��y:�{"<��<�:�<l��< �����<�۟=�< F$;@+����<\`�<�̏<��~�и{���N����״�č�<q�=�$�`�=Tļh��< _8�3�V=i��H��<������<�����X:����#=��6���-=���;��<�׫�$y�<��6<�����D�<}���|[<��<K����))V=�8Z�L��<(8��8�� |#:�<���:x�< �ջ�<2���E��k=XCk�l�P�4Ո=V�˽�-�=
�h=��=~�=�V��̒<�Q��=���:)����$=}�2=�0���s�=��/�7|��� ��Ȟ���q�C=L��<'��@.z�`����ܽ��'=м���% <�kļ��=@�Q���ǽ��������T=����ڽ�v�ܺ����߼�1ֽR/���ýV�S=��=ح�=�C��||�=,��+�V=��ͽ1��=�G� 󠹰����?=�=���Rۉ=^n �z�ֽ��=�����Y�= �P=� :=�Ur��9H��N�=�:=�rV���v=���-Ľ�=�g�����zߙ�O6�� x� 9�Q^<rgýK����ҼZ�2�x�ȼ��]��&�=�M�;��m�<mq/=��=`�����>��%��G=�
e=��4� �������<\\�£T����=��̼�`�=��4�l��<(H<������$�2�U�r�N�� ��5޽��
��
"�po�;FE�b���Z<@̃:ߍ=���<d����1
<�����Ѽ  �<�$�(~u<ċ��00/�8�<,G= �P��;�X���w�3*o=J��=��=Fh7=,s��7q]=0g�.:=��?� ����<P���v�&=xn<��<��=�I�<��;u����0=��6<x��0��;X��<��k=4�Q=�ue��!A�Ȁ���=�J�;xHJ<���;p�a� *<���؜e<��.< �W�l�<t�L�2���[�=�4#�K�H=p�j=�&����P���< �N:4j�<�c�<�Q�<0,�<@��� �Y�0f�=PD�;,��������=��X=�G=�I6��$];�Q�0��c��dO�<h)7<`4����4=p]t��<�>����y=`�v3=|����>�<�½������B�3"T=�����E=�5S< Y�;p��P��<X�u��Q���==�^ڼ�t<@x�;L����#�:=hC�<��@=�������� �h<�;�غ<t��<D���� �`���{&=t�b�l�Lx4=����=�zH=�D�=.RL=�S�:���~;�=@��y����e=Qt\=  �5U͊= ,�a\��< �ļ�<�=���< ����TH���<�#ּ�O,=Hr<f%8=�t���:�=��%=L"���E���|)��a=p��;"6ڽ$F�<p�Ƽ�	'�eA��  �8m��Y�=�� >���<`��r��= F�<�0�=�Km�|��=�
y;�I�;
����̛=G9=�UνT�>9n�Wý3sm=U�ý�r�=�q=�;�<��Ҽd�˼yU�=yR=D+��Ǖ=0�V��������<�۶�Hp�8Qż$d��P���p��<���<F�����L���|�#�:�[��`�#z>�̗</@�KY=��b=�{�=@E��������6�=<|�<Z$��΅=dp���v=���:Kͼ���=,	����=��w<�X%<!��=T:케�M�vb�B 6���g���}½$ؤ�(�&<H�c�^����u<���{�=P�=�睽 �W��(!�2i����Cm� =�;������<8�=ء����;Q1���A���[= .�=�=�=C�%=\#ۼM�d=�ِ�Y�<X]�(�G��!�<���Y�
=P_�;���<l1�<���<@A�:�Ʌ��R"=�^.� ������:<d�<*c/=:(G=l-��*+��U-����< � ���8<����
����Y;�a�T]�<��C;PsH� QO;�dk����Ά�=���� �=�k=�S��
�Y���X����<XS�@x�<D@<�J�<��9<@(�u�����=�:�f�tk����<�Qo=�=,pk���s;�J�� -��<b¼X�< z�����=D����;?<�;��[G=Hn׼���<0�$�<�<�J��������s�t�5=d[[���@=`U�;H�`�'��"x<P'��hu�v�b=����[:��4�����8v�E�=ȿ�<u�K=$ɼ�pR�I<06%�x|=�M�<���� ���wJ=��=�W� &<&I�=�1��p�%�B9s=;��=�B=J��=�ɬ�$��<@��:p��=���< ��:∪=��?�1���hʋ<d֕<�F�<e[q= ���|5�`O� R�<H�<��=(5F<h�D=xIz<���=���=�����ϼ�O���;�M� ?����9=�u��P��6�V�:-=�_ͼ��I= z >p�M�3��6D=F�=X�=������=�/� ,�B����>��=��q���=K���d�7��H=�����=Uċ=��*�Z�=�Zڼ��	>��=�G�!�=�:��$﫽,﮼��E�B�;=Q�J=t���`Qh;|Ă=���<X�+�,�C�<��<~�A���s�������=ru%�%�ƽ�ۇ<�A�<:�J=(��<�E��p��D�=�f�ړ=8T�=𰟽z-�=��9�X@u<|`==��Լ}�=>`m=�ZI��r�=��<�>����<��;��
��
Y'��S�f�,=(}�<�eܼ.R�� !�9�����S�<3K=B �k9.�8���H��p�u�4Y�<M����x��<Ԥ�<�UǼ@�G;�i��ߠ����]<��=&�)=L��<,NѼ*�9=𔶻�N<^����u< K�:�۔����<�/`�`:��P} < ��:��r��W/�\��<�M�`����Ȼ�G=h�<��=�V�`��`Gu� �4�؛żl��<�ҡ� '��p�:��v���A<xn�����p���ݼ��9��(Z=:��8�<Pt�<D\;�����к��@�����p� �};��:`�m��n�� ��&5�۠z=�q��݃�����%��6v6=�k�<�
*� ZT<jq�,6ϼ��F��'�;,&ϼ�g?��@<��`�<�Z�d��<ĄA��X;\���hT<R��\ ��h�a�(��<���#=��Q�N��p0����V<P� ���ἐ�!=Ľ��0g�;��� &/��o��̜�<�ܻ0E=v����HF������dT=h�<J��Ȝｾㅽ0U�P=M� ���=l_V��YA=��=X
I<��,=~��������<b$'�@ ���������$$�<I
.>���r4&�,k��5�=UW�-��=�
�=Pi`���<�wt�ľ&��]J=����=�,����<��-��L��.�Зu<�nN=��V��~����\<��'=@Є� (�;�?O�ڶ=�HQ����jч�$6��l1�t~/�j)|=��j�	�=���'ޤ���x=����@R<�N��
l�4����W=A'=��.<L Ҽ$�={=��j���K� ӝ<`��Xڕ=^N7�lW�PP)��ռ�/�=�.��X;<�u=��;#=d?���z��fq<�P���#���|=�A�<
�n��d���d���ٽ�%]���E=�g�<l�6�\ս=`p	;lv������v�� E}�T^����J= B��:���O���i��R7	=�&�U$��&�}��1=����Z߼f�W��fr� ���HIa��5��:|;��<�===>!=��2�@⓺`��v��Ǫ<8��@�W��1������0�p<(�X��<�!�@Qɺ�eo=(2/�.�=0��<�t��U�<�GK�+=`�;;�*K����<P����.=PN<�Na=8�!=.�N=��Y< #k�C�;=�rK<Y����=���ל}=�ea=@V
;̰�<��M��	=�C�<�GS<0H�;Ē˼�I�<�%����=�=��l�弘�c�� ����T=�#�<0��<�c�=�H<hoz��t��<�=�O>=F�<����c=~�=�k<!���=��}��ͼ@�s;�:k=�7�=L��<�*��p5k����=`� ��]<pB�<��<�	�<�&=��:�<o7�X]=�A��fV= !-�.<=���<� ���<$8�<DM�z= �<���<ܽ��s	=P����꺈��< +!; ^�:����h�޼�a�<��<k5=Τ�=�v@<�����<�"�;�:=���<�B�,[n���\�h�<��3<�ʪ�|�=�-Ͻ�?.=�>M=�e�<\�|=c��߈ڽ�͘<4���b�&�S�.0"��p�;�R�=h	$�5o���"y�L�Y=_����=s�m=Xa��P�< �M���	�^H=N�ǽܤ=vP���=��[�p�h��k���müҟ=�m�;�a����o��m�<�,v<�%û�{��\���P	r����<^<�����`=⛿�CS=���?�=�^��S���=�Y=�0C;0ʕ<����W���
�����Y�J= �ջ����P �<nL&=������S�@x�:E��`Og����=��(� _F�@��;8A���=bV��^�����<|��D��<��U��\�R0f������������<W��=̼���0o���u*���1�|=�<@��:������_=�i=������� ��f̦�P%��u;�$�;��彊bU=�>��Yv<�l��0��lּ~�_�M7��������/�y:ڽ�z����J���h�ռ�w<��;�q=��(=8r5� ��d)¼����`�m;|5>��Ps<�c޼�7�N<X� =��y;��=��T�(�U���=lڸ<���=��=��XnF=�����?=�����,V�
R=@7���=��O<�T=NcZ=� =X.<h;U��M=�+G<АH��z�<0���q=^ �= ����;#�)4=H��<Dކ< ��;�gʼ���<p�ϼ]�7=�/=�����y�t�y�ܕ�����=���<!�J=P\�= �i:ԗۼ����4-=O�#==�<`�W<9X?=!%=<ʑ<4�Ҽʸ�=�L�;��K���	���9=�K�=�:�<|�xl��ߍi= a����;(kS<��<�;���<����X�<����a=����x@=`�<�V`=�?,<����</=��g#=�ԉ<HI<�K��l��<�J�����ƻx=��R<�P;P�	��Tۼ�<��c=e�=a�= ��:z��@=�/�<y=DИ<Uw���Y�0����<0����*�G=�=�h۽Xi�<۩�=��9=�Z=Y��������<��2�6���?:<�~�<�f}<���=����ɽܷ�����ؓ���=|W=� +��`%=p}\<u~н�/=d����=�̼z؅=����l$��H�D�����	��=�8�< !T�h����X<0�<:���^����Z� 5��r�=-ɶ=�f���N�=��ýԡ=
,��6_�=Pޡ���=�x�<	�< =f���u~�ҡv�8����
�=�LB��<�*uF=�̑<��R�t�.� =0�ƻ|���vt�=t��`oU��]r=�؋���;x4���`m��^�D��@	�Ȭ���3��d�� A���d��i����= º 2D;d�=X��<��s;��숇��!����
=tҹ="��Ԛ����ؚ3� V�����m<����Q�=0L�; �º�-�<�����;{�ν«\�k��'�뽻�׽Q���J�p㻽0����P< �L����=U1G= �G�(}`� Ⲽ\HҼ�!<L�+���<�d�����@`�;�SC=`t�;W�<�]���u���=XB"=lk�=���<P@���qR=�b��D�<Vȼ�������<HV<T��< �b<���<qN/=p��<`�;x��e=8"/< Ҏ���<X��<	�=�\�=X�� HL��Dݼ�=Ծ�<�L<H�C<��@��Ƌ<��ɼ��<�0=�h��(�|<���>���绉=���<৅=�$�= �;;��꼤M����=ԩ�<�<pXi<|��<�[8=�$�<�	�_��=Hٳ<��W;��޻�E=Zi=4ҿ<�sG����/�=�%߻�ûH�m<�#s<,T����<��F�`l�<�s;��i=P[Q��=p�n<�-=���;h�0�`��;/�w=���O#=�Y�; ��:pAp��'g<<����gf<L�m=���;���;�\��@����N8<nZ�=l��<��f=�'��Z'�p��<�*�<�;�<(�k<�\����<x�ͼ���<�n��-g<
�=��m�<�k2=�<�=Ds=�����3/�3xP=p*L���^���<[�E=p�;�A=P��r�� ��:@��p:<8!~�p�<�-��E�<HR�<�'���=��<%�v=����W�?=8�0<�����=Z��09��w�=>��=�'���6�Ph��Do=b�-��1�,�g��=R~&>��=�����4>�"T��;=����$�=���<���e���o�<���<Rp½��=�����gp��1=w����<��P=@��<��a������6=�oռ��½��o=�Z�ht引K�={���^��Nݘ�H��X������¼�H�M5V�����������Y���>�m=p�<O�_=5�=�`N=��I�,�-�B���*;��=�9J���6�<��X�O��2=:�z�"=ʼ/�{��=&n=���V��=�rȽ��<�ʽ�t���5����Խ����8YP����g��)�=����t=d^=0鼽"������Z�L����H�i��rɻ��x�8�|�L>�<�=0^��l��<r�j���P� ��=�3=��=}o~= �;�Y=��j� =��$��v��<е���= �:�b=�E_= =@\f;N��Yzp=�� �0�Լ�ȗ< ��H�=ǘn= E�;hģ�PO�dc=-�<��ļ T��p�� ��=2��K��=� =0Ɲ�@����ͽM~�=H�W<�p�=���=\��<\S��O ��o=�<��]=��j<V=�T=�2�<R,E��=�ݠ<+ļ@/���hY=��=ԁ/=$G��8Me<ޑ=��{; �): t�<�PN��Vڼ|1=@/;X�<	��,�=�>�"S=�=dk=���<lj!�`�;7c�=\�{��=��Z<���;ܪ=��3<��X��U�<\�=��ٻ���\��<�&���<��p= �=�=�5�\'���|C=,X�<:�M=�?�<�R��>u=`�3<8�t<|T����<@�;�H�P���0WG<K~>^�4=`)p<����=���V�:�\b�<�-T=p��X=|�8��� ��9xk��k=�K����;p�Խ�%��$�1=��%=}O9=��<�=�I)<���<��C=ә"���ϼ�w���7�=vH�=����59�m��04�< ֋�@�%;p)�ew�=��c>����׼��:>`��;j$n=�L�á=o�#=4��&T�ON=pä;�V� >�,��,���n�<	�,��<��2=�|<(�������ku=���P���� 4= �b; cܼ�T=͸��傼����`�<�����<vL��*�EWE��/i���{�+����`��>��p=d*񼰴F=M�=�.g=$�<�
t���� ����l'=L�� !�<Y���S<
r=��,��9p<x�޼��>��= �¼�.�=���=o����w����V>�+���Jr��@�d�৛<����-=���,LF=��<8}½�����ü�d�\��[���9���l�,����=$��<ܞּ���<�ld�{ݭ��9�=��=ǔ�=��x=x�B�7Er=:���<dƷ�����v�<t��dc�<@���)�=��G=�^�<�-�t4v�"�:=�/��LR��ખ;��N<q%%=�==��l<�����Q�H��< ��;|xҼ�l-��&���J��1����E=�>�<�Ua�P�K��l����6�= ��:Ǥ]=Ǝ�=��;�u�V��Xq�<��}��yD=䷭<H�<���<or<jw]��L�=e�<��ռ_#���<FЦ=��=�����fG<x �`�;`����=�<d8ϼ��f��	=�X,��h�@Kw;���=�]����= �2=�5�<  շ������;�e|=L�����=п<C��+�����C����<3��=�򝼀}�����������;8�W=^iv=2��=�Uk�8���=��;1V=<�<�7�[o#=F��= B"��5����<r�.=�.h<��]���$=�j�=���<�P~=�˶���;�� ;@�,��<�Y= �A�⯼=��FBf���/�`�l��`�<��<�	���@��	��T=m�s=b�=��`<�9�=�=�&�=|u�=�#���q��O���z<��=D�Z��8<�[�<D��< ��:��<�����=!6<>���Xz���=fL=�>�=�X̻��=h`<� ���IԽ��
>�cG���s�E��=mZֽ�����<Ez��P�f<90L=@o��	QQ=��h���=PB�<��\��c�=𢧻�S����A����]f[=y��=L�<�����=�l�(~p���Ž�㋼M���0佌8� ��=F��ν ã<�Bs<P��<q�4=��8���5�7Y=��x<Ԧ=p�1=L[ͽA?=�A=��(<��i�±%���>$p>�1y�>> �<� l<ܭ��f�c�D�½/��I���ԝ;<Z�<`�J;���!<�?���<,W�<�ÿ��ռ�~�R�J��>)�T?��H�;vV��#��Yv/=؆<Tdм�=�<�|���н��A=�w=(��=�MZ=�pS�#�F= �q;`G�<�tO�@�B;��_;\�⼘��<H@�P0�<zf=����rμ�v�T�<p�� ѱ�P֪�t��<�z,<��=�9<8e���m��㻀�T���?�8���.���hE�.��B� =�kb<P���Bü�X���Y���H=p�L�8
=�eM=l�ɼ�X��t�����:�I��m�<�Y�<@�^��;P���<v�f��y=���;x���	�h�:�㭓=@�<Io�0ȥ<�;(���O����E<R�#��8���q;@!V���M: ��;�=H	����;���<�:�;l�]������;N�&=xp����< !�9���^�0Ż�tҼ��2�,@�=�͢�0��;�1,��*���W�\M�<LF�<��J=�F�`>˼�$��x3ռr:�=���<֣*�l)����׺(�~��\=n���ic= e�(ge�BŒ=0t�;�!J=�_����:����.�1���l<�鸽b\��9"<s0>�T���y�P=6����=ۈa��d�=	Og=��d��&�<�^��-�Ƚ �[=G���T�=0];���M��x��4P��x��� z˹�V�=0���,�����1<�L=�}=�T=ǲ����=4�Ƽ�Ɋ;�g�<s}��s<4���'&D=��	�>d����
�
=�h�;[X=bSӽO�����p<��=D<�av=ƥl�@D<��u=L�=9R}�L߉< ��L�k�v&�=u4���{< ��;dW�<�%�=��<,�<�v~=�h)<0B������`A��T�缜����[�=��<�B`�`�^�R�7�.W��q �����=�6'=(���� �=vp=޻��kF��U��<��@J�;8�=�S<D�$��F�<4�<0J�<�gm<W����R��bA<�i�x�����d*����P�����P�<| �<0��;���<�|=�qT=H����O�<��&���ͼ��<|y���˪< ���vT;�t�<=45�<P �<�v� �|<=��`;�4=p�L<0�¼�#=��`���<8_�\D���9�<|��<�0�< �<pv�<$̶<|�=�LW<��]w%=DA�<�z;���< a*:�|y=��S=�(�h�%<H�~�;�=(e�<�M4=��&<0��;� �<���:��=���<�S�����kt��ly��7-=~�=��=�?J= �\�@7;4��,��<:=�����::�=O�5=��#= K!;6^�=Pg\� ���z*<�e=H+J=첃<�?�xѱ��ґ=���@\5����;�G�<�w�<M�= \;�o�<dڗ�2!=�y]�9�=0�	�r�C=0��;({"<0��;�U�<@�к�_L=�p|<X�S<�,<��=pz���U;��b< ��;�oS<��˻�&,���<��*=Hv<�M�= �<��"��!�;`�t<�ȓ<@�c< \���`<�I���< d=����p?=��������]�j=��=̓�=��d�R�R�V^"�L�ۼ����bH�Zܻ3X<X��=d�������Hާ�L�v=�W佾V=#h`=
��z�	=�=�R�	�==w��b��=t��ܼ�"���l��b&��E,�"��=�[�=i� �Ǻ���<��i=D�=�3�����<�E;Fq6=�n�<�ĩ��L�=��㽛d=(�ڼ /> w���|���V�<�k�����=��Ƚ�޼��m<p��<@Q�; ��<6�m�`�<��<J�(�}j6��i�;�y��Q����=�0��(
 <#�b=���<�^<�ռ��=kf=������pƳ<}%�=����G�� ��<��S=�r�:��=��<<p���Yꆽ�94=�O�<�A߽|�=��.=�8���� ���H�c��:=� =���<��㰋=:T=��<h�-<������<�ߔ�&�$�Ľ������4��׽��Z��(<h�l<`{�� ��:�^/=S�O=(Tp���;`�Լ�D���fi<�ٶ��J�<p�< �;�N����&=�#�<�h<�����r;'J=�i<��,=׬<��¼Z�O=�RZ����<�����t�<ij<`8,;�y<�\H<��<�<P0�;�SǼ"7=�x�<�.;<I�<Զ�<��Z=��M=@͝��&� ��k� =�_�<�6=��%<Pm<�]�<��@��<m&=�"̺�� <Hˉ��rz�>$=��<�[9=b=��_� ʅ��}�r=��<��9���< �<�&=�_= c<N��=��;�5�;P���(��<�1=P�$<��(�~���y=����p���`�;��<p�;@��<��Һ8m<@d�;B�=��� ��<h^�<�.N=�Dq;P�J<�R[<Ni= ���g�B=P��; ϸ�`u;��<���:`t �@G=��1<B�; � ��:��� _<�TW=�2�;C�|=�<I]���}<���<P�S<@��;@��`>=s�;ƀ<I��$=hh�<ȁ��P���:y=_�=< �=�{�����fT���������g�<8)=��<`�f=�M��=W��;�<p眻 :��}�P��<�����L=�K=�8ջ��<ps�����=������Ȼ �o<�tL���
�Z�o��=h��=�Ƀ;`$�;�D4<j��=h�*<�0d�   6$�<���= Ӹ=���">D4ǽ���<��$y�=�K�=�]	�@��0S�a&�=��!�S= ����A�<Ȓ<D�ռ�޼X=L���Y���q۽���<��:�����= b;�X�)<�v�=��(<�䬼��5�`�=pL�<����?���[=��)�{�����D�{J����H$�=L�&=�w=0�H=�ʌ=(�]�ث<�c׻�sȽ�i��6NB=�e9�ĸq�@1�g�;�_7=��S;`�<h˽�4�=���= |�~�=������w='1Ͻ,w���轴
�����r<ɽk���9�<`�+;P��;�i<��3b=�`=
�%�� �ͼ���(<Hۼ��;���`M����B�y =��]<H�+<\�����w= AĺA�E=1�<��$��W*=�.����<$�����K���< ��8����qW�<Ӳ<���<���<@�y;���QQG= .i;����q=�<	a=��	=pU���'���(׼���<�Q�<4�<��d<��<hn�<�1����=j�=@�x�<&^��h�� �=��<�U=�m�=`1� 48�,���x��<h̗< 0�7�y;��<̉"=��Z=`��H�=0D<�%{��/�8^�<C�=0��<h}��nA��y�=���p�����<Z�< ��:� =�J�;@���xy��lD=����g=��<w�=X�L<`�
����;�(=�k.�=�b3��_����Ќ:<<� ���a:r�=��1;��Z���k�P���Rc<3�f=H��<�[=�x<�i�\��<��<�� <�*<�lp�?r�=��0�J<�oK�߉�=�[ͼ T�H�7���(���=��i= R;�
ڽ�^y�?,�������=׫5=�yۻ 	ӻ�ڼ�	߽p��<��A��?�=�ý�;��۽N�=lnw=��U= �9�c:=x��=h�\��$���	=�۽���u+�����=U+�=��Ƽ(�ༀ���h=@}D;8{��`��]�=W7 >�}n=��#��D>,M� ��<�(Y��\@=]��=6-� ^D�P_ἰ�<T���I�=�V����i���漿��`�)�Xkp<��:���~�𥅽��<D��������~��д�;�=`�=�����<��촋�8Շ=��; `����D0R=�}\����א��ĺ�y����@�=6�=�m=�,}=��=��=��:�B��)���V��~� =��G�@�����ƽ̵��p�=z�S�Wz���㼼c�=�E>p��35�=�>l����=ɛܽ��\���`��6�G�T����x���[�=t�>��s9=��i���X= �=ty��f@P�dם��R�G伀�H�n��>M|�X⼀�y� ��;�δ�@��<r���!Q�Jm�=��b;�L�=��=�b<�)�<Za�n�=��ǻ�Mc�r�=���8<}��*<o=�3=lj= �1l��L=��Ҽ������<؄ڼ)�T=?�< �e<������N�T��<���<|e� �.9��<@�x;{�����=�	=�*������U��ZfʽEC=�)�;�4= ��=��Z<��
���X�$��<� �<�(W=���;���<�׸<b�
=�VN���=4ϑ<�����,J5=�Ê=`=���� 嶻ܙ{=P��;`�7;`�<�a �����=�Xw<�:B�����m��=��E��D=�\=@�=�[�<�m� #�;r�=d�M�\>�<0ǒ� �;�Y���ݻ8:�� ���jn=O,�xk����`em���5<�W=�x�=�̘=@V�;�)����[=l��<,G�<��<ҏ��,�=�|�:�/�����N�V=��7�P�`��f���m�w�>��r< gϻkO��W�b ^���ϻ �:��,=85޼���9������{L���o�Bi=J�轠01�t|��T�����Y=Ie�=��<��?=���=@R��/Ѥ��xG=�)�����zνM�$=_��=H�t�b���  ?��=H%�<�%�< ^Y<�D�=��U>������hB>���r�< �W���`���=�r�@�ļ��R:�$H�|&#�P�>&��8NU�>$m��L� -!� � � �v��c���#z�4�<�ѽ��N�A� ���@��<v�<���]�x�Ἠ �=P���<=}½��<s]i�%W��!��S�'������=c��=�c<�A="��=D�'=�.=T=���^�5�Խ�a�<�!�`�����*1���=���:�~� �;V�>��>HQ�����=��:�6��=�?�Q��q��Ѣ�90��(���{g;Q��=��k����<,�ؼP=@h0;̪��@� �����T��؊��N���F���R�|W�h��<�I�8�G�<������O��=���<넇=98�=��ٺ���<�s�����<���O ����<���(%<h>��07�<�&=��e<p���&��c=����v��v3:`���u�<`�<4��<|�)�T�"�p��; N!;�I����� R
;X� �5����0=��<a��� ��PHY����LT2=X����P=�O�=П
�~C<�� �0��<�r�R�9=L��<��q�@���h�<��=�9_�=��=��	��Y6���<�uc=�<|:�� �S9`�k���A<��6��>�<z���R����; �U<8�*��;�;=�꣼��u<k�=Ȏk<� z;�� h����<�w�����< �	���y��������bp�� ��S�=@����$��o�� �úb2=sh�=��W=\ߍ�����=``�;=�=��<�34��C�=�A=���K������ٮ<B-=ʒ���9:j�>�����;Â��pp��������<�ب��L7=и-��y�=`;[���d�X�DZ¼��v;���&G ���	���G��	=71o=�a�=0�d<KP�=�u<T��o�[=Qk�T�G�>�v�x?)�O�=0D>�D�C�ڪ=ڂ<dǈ<���<Fi='/<ˈ2>��̽`�-�W5�=X�<;�=��ۼH:� @�'�\S���)�= Eu��v}�83�=�����"�$���w�ν�� ��9��;�O=��q��D=x����"���O)�ī߼�����5{���:��<�V�=�K=`l <T�<i)ҽ ���_���	����N��\����=�����v����� ���0I�<`.�= ��`<^;��	��*?<ذ3<���xT���TA<��~=pRu��1��4R��>:&	>X�6��|>X��<G@=����D���Mὀ�5:]ν�`��q=a�= �M:0���dz5�P��; �v���|�\��������4���f.��0��T�������<��8��������<�
 ��i��D��<�)=�<�: =(�9��
�<��,<�h;�W� B�;p��*��`�;�6��`T�D]�<�}h��}߼�.F���:TЌ��UQ�8~�hv�<�^N���d;��<PG���<y���'���Z�P{�������0��D!��x�7<� <�|��L����Λ���
��>�<�∼DR�< �~<4U#�Rk�`F;��o��˼Pe�;��<X~9�8{Ӽ��;Bf�H��<���<��h��D��,��	V=�0+;�i>� �.;��X� �q��)Ҽ�����P�x���T ���;��z����< �
;4����˼��==H[A�04W��1�� 0;p�y<��[��93<0�)���PaB�l�ϼ����9;��ҩ=l%�� ��;Z�V�8�d�0~I����<h��<d��<�N�Xq���i�̛�pJ=Ȫg<��3�9l=���<�
޼�U�<@l; ��9�\E�Z���h��<���\�<����=�?����I��
�=C��2�X�V�z��=���03=`���M?�=?+;���
;�m�;�u����>�p��<\��<�2�<���Á
>�����W���/	=>��$s�~C0��Q=���<W�^=P���^=5�s=i��=����}>@d��Ã��؈�|��T��<�w�X�B<�� =� > �g;�v�����;�t���^�=h>v�Ҿ�w�=���=Y
��ג;=�O�@ϻi=�
�<&J�@� <���<_��z��=z�-�8�y=pn�<(��=8�v=eba=K.i=$a�=�`:=/W����=T���xG��þ�7�ӽk˂=�x��n4�]�-=I��Z���0Ľ���=�vL=4E̽��_=�<�5�����r�@6��p'=C��="<�X,��
�<���= <���<�[򼔀E=%q��L��5�<=p��0=��ͽ����:��=�u�=� ����<{=��?=�=����< ��9T���b�=�?���	=@"N;@�Ѻ�����=�1�<�d<hi�����<0+<d潼�O�<�h.� �V�D�<�:��=<$�ڼ�)���W2��h�<���;��;�`�<`�;�hb<H�?< �K��=��<@M;\��<���;��'=B�= `V����;�Q<�6�<��<�Au=��[<��<��<Ԑ�<
=d �<�V<����ȳ:�����d<]w.= v�<�*=@�滰\�;�Y;X�<F!=�@��<�8=�<�� =��n=x<lb]=Дȼ�C<��<�?�<Ȉ=`o��ά�$����=�N'�@�x� A��l�< ��<|��<��:`�:<�pg�p��<\�üJ�<@U���=p;Ȼ��<�p<P��<��A<`.=0�Z<��8<D%�<�k8=8�9<0?ݻ�(���<p,<ȝV� �;�JT<�?=�wܻ���=x��<I/��.� �M<O�<�O< �;
/�=���<���'�<KE]=�e����һJ콽��t�=�2=K�;�:=��޽r.�ȼ�<6�j�@g<h�K����=�'���H���;��-=�1����,��v<l㥽�#;n�V=�ʰ=@��:ͦ���">:�)�S���<\��Ԅ��2��+{�=���=�А= N����<&e�=� g=��`�₼=po�<�Uz<�A��g��O$8=dMĽpu<��;JV�=�K=�"��Ё��L�r��Q�=�~��a�<��"=���=ͷ��@h��#rʽ(�{� C���=�~�� ə�Xz׽�����"�<�~Ǽ�t=!�7=
��=�d���3<��=R��=A�<�>ܽ(�=>����oͽ�=ӽ#�� "����;��<,��=@.n�X;<�����3=d��<C�����\��t㗽;���(��<�J=gv�=�������T=]�=|�ѼTF�<����g�V=�2��D��d�̼0��;Y� =Ǿ�
�� ��=�IU=й�;��y<���<��=�����ջ .:��޼F�= ���"=�Ȼ J,:�Jt�8!�<���<H]^< h:=<�ò<ޣ���<x��<�D�<ț�<0��@��;��
�p��(�&<Б�;0.�X	�H��<� *< �7���K; Ռ����<�4�<�d�:�'�<��';���<ԛ�<x <�Ѻ�XMl<��_<��<�6=�~�;D��<�Q�< *o;�DJ=n�	=�<���h	���� �b���=�L�<�3J=��; ��9`�ʻx�<�JA=��7�?�b=�{�<cP�= &/9s+=�Ҿ�\�;���<���<V3= ���P���QҼdq�=���U�`�T��R�;XE�<�T�<�d1< '0<@y�;0
�<p 9�$��< =k= T�;���<�k�<���<�R�;f0=X$@<`�;�/<K�(=�;�黀�:�,=� ;�2�0�N<4��<���<�)�<W�w=6C=6�����;L�<m�=�DJ< 啻�0�=py�<x�0<�;�;-��=��`���D���*Y��mz=Atd=��4=,˛<�ѽ�Qм�Օ<�;��J=�D�;�*<0!�����-=���n=L����l[�NV���e�<��`=���=������� �>�@����A>=��0�$�t��G׽	t6=�ʙ=X�=�#� aP;��e=`��<`P��F�u=�O=��.=�S(=���D�=D���TS�<�m�Y'�=��=zd��6G���m�,��=�D���3�=,��<��L=��@O�f 8�8R���)�2�~��޼0��;tW����s� ;�����Bf=͟K=��=Z w��Y�-��=d��=u� =�����q�=��罿������W���탽 3=-s�=#&�=B�=i��=T#���?;`>Y��$Ž�R��J��`�����$ޚ�V97=/�5=��a=f� �뤮�4z=��
>xLL�+�=�x���k�=�E����ʼ| C���y=��^=����wG�t��=,��<�c�<x@�<�94=:�=�.��19��̀�~u:�lݳ<�H� $s;f�Z�����X� g�9��=@�E;�8H�z�u=����3&= �&=q�-=�;<�5��S�<�ڼ��F��V�<Dʹ��;�(ƫ��E=L��<�//<�ٱ;�"��R�@=@��:ļ>I?=P� �$k=�2�<X؇<��W� �켘�<��=dÚ< J[;p�D<Hx<_;����=.�Z=(�o���q��"]�N����Q���=��<p��=`W�<�f��n�A��bL<��=��<`���`\=�x�<g6�=��8�޼�=`U����G��Qe<�2
=�ӆ= Iֹ���|���/C�= T; ��PK�;(���v1=T8�<�=�q��@	b��
H= �2|S=��n=�:/=<��<x�<�8�<P<�<�E���_�< �F<8�><h���X!�<��%��5~��M{<!F=|E�����R<��"=xj�<�=Hg�=�5=����>=�V�<,�h= �h<���LB�=h�_�@��<L%ڼ�=���� [;����ʡ��=� =�6�<���Z|y��a���6�<��<��==lEǼ�~G������Q���%/=�Á��Z�=���X_�<�.���<Wyt=&�>00�ϩ=nm�=޴Y����6�=�#�����=!��Z�<1��=ĕ�<h",�$�����(=@�7<}�<��-=l�=��=(��<(�м��>��Z��c\<��4�X����=��J��_<�ǎ� �<�ث<�(*>n~<� C<�����[�� ���������N�u�p�h<0ݺ;ν0c`�JCG��廊�=�C=�X�i[̽t`�R��=�-=�Z=�#ɽh^�=��!��6p���I�����K�ӽE�c=��=z6�=��U=��=# =��;t��.D��|�ʚ�Ha+�4N�<�� ��o= �a��|���"k<q��=�
>�$5��=0&W<�s�=�?���`�.����D=.�<@� KA��S>,-��Xr=t�<��C=��<\i��' �� gٹ����h漘��*��ν`Z��-U�Z��`��U[=0�.�v�{����=��r��=�ۤ=��K=�K�;m�����<���T�`��+5=��x�`p;<����Q��=��M=�!�<@�?��\�kP=dpܼ�O� �/=�-���@$=�L�<G�=����Hm��< u�<�ܼ�'�� �9 d��C���g�=q�T=vx.�M��4m��4=�����L�<(z�<ҍ>h8�<#�%��$s�<��=��=Ю����p= u�<�+=�ͽ�d�=�� ���-���绘�I=K�=x��<ۛ��p�;nT�=�7�<Ȳo<զ<�m(����<h-�<��= ����8��n�=�c�7�=� �=e�B=��=\�����<@]<&o����i<�`�<�1�<2	�� ��|0��~S �y�^=#�=����>� b���,=,��<�B >��=�\�<HD���g=X��<�\�=���< 
����=p���`$��cH���=����8W�< ?�� /����>0�m������a�,a漦{`�&=P�S�&=��T�R�;�����"B8���_��^��Sf=i������пq�P�.��e=1>��ټ��O=���=�b���)�X��<��Խ�j�<b� ��7�7�=D�м`c��8�V��//<�@<�X=c�I=t-�=�>P�A��ٻx">�O�;�'�:�H��h���9R=�����A<�/����� �-;�O>d�
�|�M�S}�\{��ˢ;��� q�g��~׺`&��)���.R��Ļ�T~켼�H=�`��n�gв��>�*�=(x�<l��=�ƥ�FC=\F���x�c�T7	�������=���=��H=���<~ذ=4�={�?=v��h����L���t�e����peU��s�����=�}E�ӿ��j=X��=r�=(�����=	�<)>��ѽ��� ���<û��<����L��<�N>@�-�̄�<�P,����<�]h�.Y���.d�@�f��c:��W���'���"���ώ��&�P$u<6�L��� ��,=0ݻ���^M�=nS���V=��=�_=��#<JC�0F@<l[׼̝Ѽ���<�h��2�<����~�+=|R=�?,<��l��q�̒�<��!��o
�H�< �>����;�?;�=L嚼�鼘�3<x�8<~x��*��J�@����Ɂ=�N�<�.?�uԆ�X��Pth��dڻ�E>��1�;~��=��߻B�=��;V��_<+K =��=0�;P'<��ɻx�t<Qͪ���== �<(F4�������<K͏=��<���� V�9!#=�r< <pXu<�]W�8[��du�|h�< A�;Ќ�;��=�5�@-�<&��=�Z�<��;����8A�<0D�;�׎� j);�F�<�9W;\��２ԃ�b��ҭ=���<���^.J��ʚ;��<t�<1��=j��= 0a��.���=��;��=��<l`���>�=hC0<
���3�X��䔀<i?=fØ� fE��>�wW�WD��P$��p���]��Q�d=��%�^Gn=(���(:=�,�`�(��IR��#�0塼p#u�zo�hY9�
����=��l=x�,=��P<��h=b�����X�<�/����g������<��!4=p:�����+�<�i�����:g�=�xX= ��;��=\4��|aƼ��Y=`�;(� =t�l�?���̼V�d�lj@�I�=<Žh�\��cX=f�+0�	����8���<T����[<�G�<��`��S<��f�,�+<��1̂�xP-��@�� ]=(UJ�c=%=hU=�1X=�ý�I<:C潰5l�����	��K`= ����.�TKռ@0�Pu�;B��=�`���k=����b�L�̼������ؼ ?�0�W=��	���0UO<�/�=��=��Ӽj��=�=C�=KB��-��0۽����ͽ�g�a]=��B=�/�<(�(�\?�p������,������K��좣�hؐ�������;@@!��Y �$�<��>�(GI�P�<��;MD���X0�@�:@n��yj_=\q�< b�;�<�#k�Nb�P��;�@��Q��� y����� 2M��<eļ�g�P�b<�W� |м�+<мP6<�`$~��=h`� ��<@�#�x^-��e�:�"Ѻ �ӻ��H;�KB�P�;�rP;�_������5�<���; )S�`$O�p[�0Ҋ� dK��*���*;�m����;O�;�Zu<��	�d����<z���a����:�(���x�l���S���,=pC��|�����D�T���`%�$⑼����B�ؒ���S�hV<(%
<�-=8����h��,)ټ���=�e�;��/��%��-<x�1���5��!g�p��;\����s�����0���"���X=�<�< ����0�\t�<@絻�dA;��<|��<�n��8�N�� o���y�H=�<0qL��=�p�<.�=��4�t��< ����<����`I[�p<p<����J�&��=���{�lR�=wf����-���߼�=(P��Q=�?P<
��=����Ѓ������C�<t� .�<��=�s�4�`�=�=�*<�9��E6=��L� /�"�)��\�<�ʥ<���=Pe���#=�m>=���=Ҙ���
> 
<Ro�ړ����a���;.M���Q<y�%=�k=�#�<2VD� ׫���)�AC^=L����X��հ<�}�=Ͻ�S~<�н�e�,1.=\�<J>����;6u��PG��y8<��+�x=�<iK�=���;�xn=U=xE�=�]�=a8����S=,�潼Oɼ\���/惽�>�<�0���ۼc�=>B����Z�̞a=��K=$�p��l<���܎����jP��(�<�
0=���=�B���B��@+�;@��=��;�0
=��6<�= �9�ļ?��=�!�洂=R뫽��7�s� >zڇ=`U><�)�<W�=P�:=��� �90to<����tT�<ˀ���2=j��G��\ �x7<�z�<(o�<�r��X!<�M<J�f����<�`л�<��6�X�$��!�;n���G�#��`�$;Ќ�< �ʻH��<p��;��X<�x�<������<� =�,����=��)�Q�<��=�W���|+<�:�<��: $<�;k=��;`�缈0�<��;�/k=$��<�d���5v����x�ּ�mk�:+= Թ��X=�~!�𰩼u���$��}�f=����7=�K�< �/=,1�(�<�Q�@�����5=�&�;�G�=��~�,Dʼ( @<� �= ݋��̝� �X�tʇ�`H=` ��8� <�}<�犼x�[<T39���<�5�;l
�<P���
�<�4#<���;h�$��`�<�P�<XE�<��<b_6=�Z�:XGs��Xw��cC=���;�<!�8v�WQ<��c<�I<�=V�=��pL���λ���=X}�<��\;Zs�=�<�� ���5;�C�=�F��Q=�y������t	�< ������<���=�����P׉=�Ѱ�\"��t�!���;04
����A�<��<��^��ｲe-���q��.���=fj">��^��s���>Ƣ3�v�|�R�=���- �'�ڽ���<�u8=h�=�*���q<Z`=��O=�2����
>^��=����"����
S�0��;��＜�<Hw?<��=y�H=W���@�E;|*ǽ�Od=x+����$=`�<�i=�P	��L�I�ֽRD{���<�&���zp��D`�u��py�@I����O�= �Y< ��=ʘa����<
�=$1�=� �=J0�$z�=b�*�J�il���9Ž�a ���;��}5=Z��=���5W<�Y���=Lk=����a���k���Hɽ�ŽX����<�D=�Ĩ=5L��U*��P��<\�=� :�
=�=,Y�=�����7�e�&=o�!=&��=2��g��.>C�-=<��<�c�<l��<�t�<�0��<��hx�<�A�di�< ��;�#i=�6� �»x�-<`\���u^<3&=�ر�`m;�N�<�$��H�<x|<(^=�
#� <�� ˺��@뙺p7�� ����O<�>��s�<ptC<�-���<��<O�<TE�<`{W�&=�)9����;�%�<��<��i; 6�< ���d��<D�<XK�|7���x�@Q,�Y�=�f<=�&5�ပ������D��g���6=���<U%=���<��@L�,�����=�7$<�%����=0��;$FZ=:/���:��.�x�$<,�A=�/:i��= C�������<%�= �l9�H� �i���0�t�=HV�Ȋ�< �u<�Nd<ڤ<��p��<*�8=$��<���h$�< ��<|��<�k���gc;ޥ,=�S<�:�;q=@�;0q��P�;J��=�<>!
��#�:���< �4;�� =�ɕ=Y+=oD�09�;�:���=X+|< !d�Ҩ�=0�?<� A��.�`p>P��5=�5��t��<0�><�eZ=�M�=0������B�=�"p�@;��)�����;�J{�ޮ�=0;�<�<����q��er���,;���<V�R>>e��Ȍ�fb�=�O���P]����<lJ��'��B��@�Z<?F=1]�=�!��`�W��HE=|��<pu<5/>�X�=��{:ĸt��� q�<p�<-d==`<Ļ��;�"�=��;� �;���g�t=�A<��= ���,=�s��?߽ާa���s�X���z��0��;0C��(a� RL� )ȺTP����=p�-�r58=I����<���=�n�=��=����,�=�tȽ��_����`���J��Ќ�=Ɯ=��<�Sf=�J\��z ����<�ힽ,���R�$����Bg�df���<��>=CG�=�ý��޼>� =�B>bn���d(=G�h=h��=d���*:N�X|<���=��=�ϵ��7F��RA> sT:r�-=�,�<�@=�\
=���������<Z|k�0��;@�,�_�<m�Ͻ䝈����L��H�z�"Ί=p/C��e����=;�,q<=�<9s�=<�ʼ ��L��q�4�����k<�u�����<�O���|=-�
=��J<���;�����I= j<�2�ZP=@s����\<�&=8�=�"�:��Լ�cݻL2�<�\�:�����d����ɼ��I�2��=om�=��r� ����PEM�FU���n=���<�j�=U�D=> Y���?�xf�t��=�=xt�l��=�\�<�+l=[�̽�<(0J� 	�59.=�u�<�u�=th��
I��4=�>T�<p<�FS�C�����R=X�N�@U=@�<��j�;=����\c=��=�EF=�ޓ<�V<�d)=�D<�U� (��V#r=�ۓ<0f׼���<0�J�tS���+=���=;��.=e��[��SS=��ZZ�=�[�=�\=�j%�5\=0<�;�	>��<<��� ��=8h���<�N��x^'>$�ֽ�7=���b��ϥ=`'�;M�=@σ=�ͽIr���(�=B��0����/�
�`������=�=�R��̇p==-(�Z�4�X"�<P�<b�=�=F>�A�� C�<���=�b���D��@k<\�4���a<h��������	=6�\=t�\�ؼXU=�EN<4=���=H�>�e�<�F����;�Y�=�cR=�A=�𭼸�O��җ=m`�1�+=t�ֽ�bY<$�Q=+!>2B=� �Y;���u� ��^�=� 8�&��6�7= �J9ڊ�d(�< K�����o��=������<����bм1Э=yh�=�*�=_w����=����:�{�wT��_9�0��<��=Y�=�I+=��=���<��U����<��wG��uǽ����a�)=_W<�J��"rV=1nW=����Cl=qV=�s�=�l�@�^=pM�=.�>��������쮼�L=%^�=����pu��	=>���Vh=(��<_v=df�<�<���ﰽ^�<�5���g��̼��ٺ����ݼ R��.�м0�HO�=H�μ-O��5z�=D�!>�=XÅ=wt�=�;μ�V3���<�W��XyW���=6�V�!T=�+��!�=�,v= X�<�K-���.��Jb=���:�/u�ji9=ZĽ�<� m=��V=��#;P�L�� �;���<dμ�����.;
�+0��8�>R�=6�{��|����镽�`� K=��< ��=�>C=)���'c��P��;B�=^�=(���f�=(��<�k>=���M�1=@��,㦼���<[x=Tu>���>۽(Gm=�Q>�=�6�<�_<�^����=8�	�Tlp= �?<����=����r�=�|�=��v=x��<����!P=�,B<������	�Ќ=d~�<����	�F����c�7�=�\�=����)��TԳ�Wyq=@��->�_>�== |�:6ȁ=��;��>�J<����L<�=�AӼp��;<�����=�6��N;=��A�@q8��/`=��׼8�O��Ҭ<��@�!���~�=� ���l������V�\��� Ӭ�xZ�<�7��Z	=����6'���=�@!;.�A=��!>ܷ���K
=j�_=����qO� 2N��w���]u=���h���=�<�z���� _�; �"<7(N=���=��	>Nr==������<d��=�/h= $�<��@��!�<Ȱ<齽�V�=ڭ�����2�={�=��۽����h:�0P$��g<�9��W	=�}~����<��b��`V�<�!���^�BG�=x���=՞���7�v�W=|M=�C�=n�&�g:�=�n
��&=��?z��5Ͻ�"�=�[�=�=��<�c=a�<T��<��}�p\�<�sɽg����妽(��<��=^7ɽ�4i= (;�+�xg�=r��=�k�=,C���=Ԣp=�>&�N����������ؐ�<h_��&	
=OX">Є��F,=�1�</=���2�ٽOV�� s�;�du���,�T���x��Y�T�7���|<�7w�@V����=�_y��Ƚϝ�=�D���b�=�=���=�j�����<�l��5;��=�)r�he�=H����=J(�=@r <�P�����z�=<z��~,���< ��`�X;�I=#�S= g�;�'#����;<�<�O�<"��6��4�m땽f�>�o�=cc��p��h���j�*f����<�{�;Vy�=8�o<�菽r�v���x:���=���=Ц	<��i= G*;\�<lE��m�<nƼ����b<�O�<��>�n<\�ѽG�1=ߐ�=��=��<(9.<��0��;x��3�+=�}�<  F���Z=X�:�W=+�=D>=��:�P��E�X=���:҅ݽ0�/�dO�=���<����ؤ��>���1?����=� �=$�C��ѐ�t7��ʀ)=P0ǻ��=�1�=��c<��<hY�=Pi�l<>(lG<��ʼ;b>= ��:�OV��r׻���;��߻Q>l=���ս~\9=�	Z�ΑE�h4u<�����"�=�-f����<0=��OO<@$c�|�=�u�P�O�2�Dz���h��#�<���g=#�= �� L�;�/�<~�|��3(�`>G�DKW�~6=��½�}����;`gB��D���;������V'�0<=iP�=�^7=@k<��޽�$X<��<��<�= 8>��Y���"�E���,��< ,�;���@p�m�<=D�\l����н�L�`'�;(ɼ�'3=���< ����k<𩄽|{�<��������t���Q<��p�=
e����;`&;At�=�u�="�h��:$=~ ����=�9��Ԏ�LqǼpi�<x��<@�I�8(����� �j��\{=�۠��Ŧ=������ܨ��1��8s�<�F��>�=��f�1���=���=
%@=�ϼDx=�P==P�=�	���Ͻ
H���n���/?��σ��r~=���=P��< ��<�̐<x}><@���-��bL���;��C��q��T<K�T��<a1���G9���<KZ��x��N�w=��<�F����<b�^�\
=�F�=`�x=����>
��%�;.�h�H!B<�=��V���=D��� �+=�c=@����(��u��Eּ�&�������0P��7��(0q<0�'=�R<�!&�@1ʺ�<�C�����о��O���\C�D��=��<=����ҽ���<@뾻��P?v<�j�;�(=0@���F���u��>�ʴ=�Đ=� ~<p+�<,�pb�<Q�˽��@I%��'ϼ̺�< ��.��= ���$Dj�h�0<{� =��<�gV�`.����  �4TL��x�<Ԑ�<�h�<�Y�<�d��gN<���=�¡<H㨼w¼iR= ��bx���ہ�Uu=�_�<ZՌ��]�;x"�&�z���=K�=8R6���f� ;�:t �<�H�-�=5S�=�j��@��:=p1���B�=��<LȆ��B�=��=��7� �M�j�=BK��w�;��k� y����¼�h����=q]�� ݜ�s�g=�G[�:o�8��(;%<��a��=��4=HD=(e\��q��^��cV=� �@�:�=𔓼а�<�p%=d �<�[ӽ��*=l�R���<DX��t#�<`#�<�6!<���<`��;��<�=�y?�m#�=>in=�o�������;��K�<(��<�̴<��<�7=`~�<4���8j��+u� J4=|���ذ����<V�=�M������W��{��cl=H�r<�4�6'=����V
= ��;@揼���<�<�=�xļ��,=P��;��#=�m�=]ӹ��
�<����l�; ���H 0��<:9�������c=8r� N�; S2�(V�<K�2=��~�=r�Q���F��, �������;�v<�=�6��X�<�xd<B[�=8<���=��Y<���=h9< 6��W�=@'��x�B=<���+Ǽ���=H@�<���<|:�<��S=��=�c�����E�<����<��м��g=&(q���(~5<0��; {�96gJ=j��Ht����<�K���(=��;�= Dh�0���Z<<�6$��f�;�N���T�s3=�۔�ۅ=l\�<hY�<4��<xN>��6=8
=��ɼd�?=����+=V=�r"��3�< �:R<h�&<>
Y=@V���(���<�g�����=X@,=h�ӼG��P��`�L����JB=�(<��^=@m��t�Ԉ�������~�= ��Ԓ�*=I=���<0�$=�@��0p�< �r��;��L�B=�q<��=�N<����m�<ڢ�=�6ѻ�΃� $�;D{꼄{=��-�D�<���<�8P���=��U��r=�C�<��A=0.���v�<<�<0�<H߼ଂ<eCH=d��<�\<^�= ���x�t�P�.<�"=0	�;��A�K��d��<�VK<���<�ܾ=-="� � ��8!��ا�=4Y�<�j�;O
�=�g=h���2�f��=�z�&�=��`ѽ�m�X�ü�(w<D�=��b�w��/�=���=����)� �)��x�p^[<��2=�˔<H��ԓ�rh1���q=2J>�`}����=@�g��<L|�=��.
=�p L<
����M=m���җ<z=��<�Z��X�'�L��<d7�<�!��҂�=a-�=�n׼D�𽀙9����;tQ=`��<�<�������<RJ����;>���=��;�P�<��<�j=/��{k��_��2�>�_>= GB�$�� ��:
�����=�7��l�i]E=@� ���B=~�l�=8�d=B$�=���=�oڽ�b=T���I���4�v�!�����:�M�<��<r��=`��;�aK<�E���}�;e�H=����uýַ��8�%��jҼ��z��̿<�6=��W��9<D��<��=��ݼZ{{=�#=W��= <[;�&�S"y=���<�e�=<E��,��B>���<��<�<�	=�=P�O��c���'=����P8<���:>�=VZC��4���=�R�;���;�vb=`CO�����w
=�t'��P=��<ȎV=�	�~�=�;r�����<�~:��g;�E=� �|�<@�<���;pKK<H��<���<�=�Q���=@w-<�w<��==���<�<T��<Pz�;@6$<���<P�r�paؼ呼P�;b��=^�v=@⇼�3��Ė�< (h8�����\4=���<�
=�0�<J�%��߀�Lz��j:�=4�=PU�����<���;��P=A��K����@ƽ<9j=�~�D��= #<���<�!d=��=��G< �� �:�~i���<�߼ =m�=d)�<0&=0�⼅O=#�s=i�=�?u��?Q<�.2=�=`�Ѽ �k���=�B< �;9> =��<<�\<	/=�Ǩ=lW�<�X�����< RK���>=y|�=~�= ��:�T<8^��0�>���<� �;���=tĹ<t��������>���X=<N��B�8z(<�$���<��=uF��QGǽ&�=�����{�HԼ0�ֻ�D^�0l���=p;�� @��<�r�>���p=4�޼ b��.�!>vn���U<�Q�=�H��=���X��*�U-=Oi��;�<\=��<0"ֻ�iX��5=���<@$�<&9>�>0�3�V^ս�]b<���;9�n=Kq=��;;��(�ы,=��$�<L6ڽ )=v�<f=�=Pz<��W=�-��'νֿ�$U��= ^^�@&P<ЅA������pu=hP{<Pj��@]=����Y =������=�`�=a�=�>WMĽe*�=^��� ��9WC���~��O�X�m���O=5�}= '�<���<��6��ڰ��4c=��5���p��볽�4����X¼�{*=�m�=뺠���=hZ=/I�=غ ���j=�=l=�7�=@��:����A=D�0=A��=<�/���¼ˈ%>���(��<�<n�=˗=��'��S�� =����;���;z��=.�v�����j�< �ۺ�<�:��g=`Ԅ���꼚<=�3t��,�<�!�<�N�=$���^`,=�ρ;�v�<��<�		�@T��\Cf=����B�=`H�<��~; 3<�d�<�S+=�� =�A��R� =@��:��;��Q=.�=\��<���<  �:�n<H<ϑ�6ּ���`~ܻ�4�=Π�=��u�͜ǽ��< T�8�����I=`[�<
� =t�<2�8����`�����=k�=���d�= �
;�	\=�I���D��di��|�<�O=H�Z�tv�=H�%�� <�C�=瓏=��< &W;�,U�q���<���Cs,=�j=�D�<���<@���=5��=o}=`6��P�c<�C=�d�<�  �@�����=p�;��!�T��<��l�x�<_bl=���=�?�;b�0��Q�[O(=����+Up=���=�}.=`��;�·<輲��> y:<�7�;ܳ= d�;�<@ �:��
>V�����< ��:�a�dL�<@�;���<f��=����鼽8�=CՌ���n��缨���,B��(��0I�=��ּ ��<$����ټ�u= ��H�)��� >`ϋ���=��l=~h]���#��Fݼˏ�B�b=i�����<p҅<�=��`_��J=�Č<�W"=`%>�M3>x�<2	��ò*=��=_��=���=���:Xb���^=�����;= �Žl��<�=	w�=6+�`�d<���-�� F�;ʈ�ukD=hu:���#=X\"�4M���=ɋ=(!	���p=8���0b=2^��q�<q�{=�׊=�L
>�)g���= ���Y�<�����¼7���\��< ��=�`�=�,:=T1=�;��2�w=X��<c���ɹ�NG����=��;�r-��EX=�=�P��
@�=\q== �c���^=�k�=:�=@��:J���<��=��=д� 0���V>�.��HQ"<�J�H�=
@=�i���+��`Z�< 3^��K*��S;J�u=�i��sٻg�<��<@��;~�_=�Ă�����Q�<&S$� J_<H��<�xz=�\���z=p�B<V��U�+=�Њ�`	(<��U=�t��@��;a�< DĹ �p;I�=z^=TG�<@ݎ���<��;А���f=��=���<��<�K;8�:<�W�<x7
�<͘��!�Ѝ#<�_=A#�=p��E���%�<X1]<\��b1=`_�<�?�< <�;B����v;@��x��=���<0P6�h�<��E;��F=��Q��*��������<�((=�����= 7���Ts<Bj�=.1?=d��<�!< ���^� �B<���.A=��<�=��(<P�㼬��<,�v=�R=XP���
�<Ѵ4=��<83ݼ�Y�+}�= :g9p=�;��<Ќ�; �7�`��=䎹=H�<Z�������4=D�Լ�=�aa=ڍ(=8Q1<��<H��d� >�� ��;���=��!;���<�[g<�{�=h:����<�=~�&���< *�pk�;�'�=v�F�c]����>�V���h�,'p������M��R�<�=���اM<��ؽᨼ7m=��0:���<��>�S^�&G&=�L=
-^���#�������>%�=9�׽ ɲ:��t;��<0=������|k�<p�<��V=M >^0%>�h�<Rͽ�i=�sh=��=�4=@z��,�ɽ�=�<W�(��=�D�,����y�<�2�=_����y����[��`9<`��"�=�;��f<�3�;�սD�=�"�<�|�(>�=���*_=~�Y��K;�}9=4y=��>��w�n��=,��A�5=�|����
�J�W����<p�{=坍=���<���< |�8`��;#F=^CG=ȳV����d��<�C�<h��wI=�k2=;�ͽW��=���=<�= 6];I�=jR�=���=���i<����@s��穐=�����<�><�� ����@ӷ<�yl<D��̝���;�!�������Լ��<=�dG�����d��< R�������>=��
��X�x�B<��ϼ�;�<"
=��Y= �۹�C=�:�<6�"�/88=*𻐝��Ȑn=����O<� =�$�|���h�<��<P�9< @-:�ͻX��8�E�=c=,��<d��<���<��z:��V� .<���Lۼ���� �:��s=K�u=�D���M��l�<�<�<$�޼���<��'< `�<��H�"\+�`VQ< �D����=V_5=@l��̓<p�C�{�=^	q�$Ա��HռH<�G�<��0�o��= �� ՟9��h=��<d3�<�x; ��vf�Н�;�C�1�<���<���< �|����h6<�`X=���<(7���D�;�,=x-�<��3��[��d�=��;�Z_��	�<��u�k��u�=���=H�5���� aI�]A=&���	=��k=`h�<�6<`< ]/����=��";L��Q'=���< pu<��< �<�����0=L�<@>ڽx��@�ͼh+Z���O=�\��7��%M>�P�������7��p�<(�3<��M= ��|@�PV/�Dt6��O(�� ==��7����<��=%�H�<h;�<�+H�jp�����}=�䧽�D����Lޓ<���豛�@�0�P}�;��#=g��=O٣=L����%޽M&O=��;\'=9== 4�^���8���n����=��o;h�A�8h���}<��e� �g�mӽǿ����Q;�<<*�=�R�;�Z�;��<L������=h��<L��hoZ<ԝ)�ۼ�=Z1�$<�<��5�ی=�I�=�t��~X�=¼��%=����s�`��� �<��)<��<�; ͼ�\����<�0=���= `E�6Y	�X���,�Լ��S<�ƛ��I�<V7=q½Ƹ�=���=?�9=�l���C=+_=���=�M>�z�l�H�,��c�P�<��/�!� =�KX=�9�< ���8?<@h�@���XY+��{?��,b<d��4�d���e>=�\�\#�S�<�*�0��x:=؈C<N�F�0����?¼��:|�
=S�Q=�鳻pf�<��D<b�?�pO/=p��;�3����.=`n��,<�=�`��<�� ��<Эٻ��ٻ��!<<�ļ ����Nڼr=ȁ�<lC�<,�<p4������@6;hxR�\Z���i���:ʅ=T�1=	��P���m(=J�-=`�Ӽ�¤<�|��@�O;DH��6l����<d���h˛=��Z=@�; ��;X[��j�<��d��W� ��@�c����<�&�b��= �� �<"�= 8�0<����f���y� ��;Jlb���;�&=���<@M���iۼ�E���ZX=p~�;�Z �P|ڻBJ=��	;@�B��aC�0�k=�Z<�s�!n= �>;��#�hrY=���=��˼2N����L��<���`M�<�B=PI�;H�<`�;|G��w�=���;���*
dtype0*'
_output_shapes
:�
�
siamese_4/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`*
	dilations

�
siamese_4/scala1/AddAddsiamese_4/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:{{`*
T0
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
%siamese_4/scala1/moments/StopGradientStopGradientsiamese_4/scala1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese_4/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala1/Add%siamese_4/scala1/moments/StopGradient*&
_output_shapes
:{{`*
T0
�
3siamese_4/scala1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_4/scala1/moments/varianceMean*siamese_4/scala1/moments/SquaredDifference3siamese_4/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese_4/scala1/moments/SqueezeSqueezesiamese_4/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_4/scala1/moments/Squeeze_1Squeeze!siamese_4/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
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
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_4/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_4/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
�
Nsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
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
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_4/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_4/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_4/scala1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
usiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
�
Tsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_4/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese_4/scala1/cond/switch_fIdentitysiamese_4/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_4/scala1/cond/Switch_1Switch siamese_4/scala1/moments/Squeezesiamese_4/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese_4/scala1/moments/Squeeze
�
siamese_4/scala1/cond/Switch_2Switch"siamese_4/scala1/moments/Squeeze_1siamese_4/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese_4/scala1/moments/Squeeze_1
�
siamese_4/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_4/scala1/cond/pred_id*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`*
T0
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
T0*
N*
_output_shapes

:`: 
e
 siamese_4/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_4/scala1/batchnorm/addAddsiamese_4/scala1/cond/Merge_1 siamese_4/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
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
 siamese_4/scala1/batchnorm/mul_1Mulsiamese_4/scala1/Addsiamese_4/scala1/batchnorm/mul*&
_output_shapes
:{{`*
T0
�
 siamese_4/scala1/batchnorm/mul_2Mulsiamese_4/scala1/cond/Mergesiamese_4/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_4/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_4/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_4/scala1/batchnorm/add_1Add siamese_4/scala1/batchnorm/mul_1siamese_4/scala1/batchnorm/sub*&
_output_shapes
:{{`*
T0
p
siamese_4/scala1/ReluRelu siamese_4/scala1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese_4/scala1/poll/MaxPoolMaxPoolsiamese_4/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:==`
X
siamese_4/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_4/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_4/scala2/Conv2DConv2Dsiamese_4/scala2/splitsiamese_4/scala2/split_1*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_4/scala2/Conv2D_1Conv2Dsiamese_4/scala2/split:1siamese_4/scala2/split_1:1*
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
siamese_4/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala2/concatConcatV2siamese_4/scala2/Conv2Dsiamese_4/scala2/Conv2D_1siamese_4/scala2/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
�
siamese_4/scala2/AddAddsiamese_4/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:99�*
T0
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
%siamese_4/scala2/moments/StopGradientStopGradientsiamese_4/scala2/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_4/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala2/Add%siamese_4/scala2/moments/StopGradient*
T0*'
_output_shapes
:99�
�
3siamese_4/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala2/moments/varianceMean*siamese_4/scala2/moments/SquaredDifference3siamese_4/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_4/scala2/moments/SqueezeSqueezesiamese_4/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_4/scala2/moments/Squeeze_1Squeeze!siamese_4/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_4/scala2/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_4/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_4/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_4/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_4/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_4/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_4/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese_4/scala2/cond/switch_tIdentitysiamese_4/scala2/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala2/cond/switch_fIdentitysiamese_4/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_4/scala2/cond/Switch_1Switch siamese_4/scala2/moments/Squeezesiamese_4/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese_4/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_4/scala2/cond/Switch_2Switch"siamese_4/scala2/moments/Squeeze_1siamese_4/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_4/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_4/scala2/cond/pred_id*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_4/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_4/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_4/scala2/cond/MergeMergesiamese_4/scala2/cond/Switch_3 siamese_4/scala2/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
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
 siamese_4/scala2/batchnorm/mul_2Mulsiamese_4/scala2/cond/Mergesiamese_4/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
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
siamese_4/scala2/poll/MaxPoolMaxPoolsiamese_4/scala2/Relu*
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
siamese_4/scala3/Conv2DConv2Dsiamese_4/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
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
siamese_4/scala3/moments/meanMeansiamese_4/scala3/Add/siamese_4/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_4/scala3/moments/StopGradientStopGradientsiamese_4/scala3/moments/mean*'
_output_shapes
:�*
T0
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
!siamese_4/scala3/moments/varianceMean*siamese_4/scala3/moments/SquaredDifference3siamese_4/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_4/scala3/moments/SqueezeSqueezesiamese_4/scala3/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_4/scala3/moments/Squeeze_1Squeeze!siamese_4/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_4/scala3/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_4/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_4/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
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
 siamese_4/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Hsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_4/scala3/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Tsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_4/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
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
"siamese_4/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
c
siamese_4/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_4/scala3/cond/switch_tIdentitysiamese_4/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala3/cond/switch_fIdentitysiamese_4/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala3/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_4/scala3/cond/Switch_1Switch siamese_4/scala3/moments/Squeezesiamese_4/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_4/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_4/scala3/cond/Switch_2Switch"siamese_4/scala3/moments/Squeeze_1siamese_4/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_4/scala3/moments/Squeeze_1
�
siamese_4/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_4/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
N*
_output_shapes
	:�: *
T0
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
 siamese_4/scala3/batchnorm/RsqrtRsqrtsiamese_4/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_4/scala3/batchnorm/mulMul siamese_4/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_4/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_4/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_4/scala3/batchnorm/add_1Add siamese_4/scala3/batchnorm/mul_1siamese_4/scala3/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_4/scala3/ReluRelu siamese_4/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_4/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_4/scala4/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_4/scala4/splitSplit siamese_4/scala4/split/split_dimsiamese_4/scala3/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_4/scala4/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_4/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala4/split_1Split"siamese_4/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_4/scala4/Conv2DConv2Dsiamese_4/scala4/splitsiamese_4/scala4/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
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
siamese_4/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_4/scala4/moments/meanMeansiamese_4/scala4/Add/siamese_4/scala4/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_4/scala4/moments/StopGradientStopGradientsiamese_4/scala4/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_4/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala4/Add%siamese_4/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_4/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_4/scala4/moments/varianceMean*siamese_4/scala4/moments/SquaredDifference3siamese_4/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_4/scala4/moments/SqueezeSqueezesiamese_4/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_4/scala4/moments/Squeeze_1Squeeze!siamese_4/scala4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_4/scala4/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9
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
ksiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Hsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_4/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Fsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_4/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_4/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Nsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_4/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
c
siamese_4/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_4/scala4/cond/switch_tIdentitysiamese_4/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_4/scala4/cond/switch_fIdentitysiamese_4/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

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
siamese_4/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_4/scala4/cond/pred_id*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_4/scala4/cond/MergeMergesiamese_4/scala4/cond/Switch_3 siamese_4/scala4/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_4/scala4/cond/Merge_1Mergesiamese_4/scala4/cond/Switch_4 siamese_4/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
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
siamese_4/scala4/batchnorm/mulMul siamese_4/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_4/scala4/batchnorm/mul_1Mulsiamese_4/scala4/Addsiamese_4/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_4/scala4/batchnorm/mul_2Mulsiamese_4/scala4/cond/Mergesiamese_4/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_4/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_4/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_4/scala4/batchnorm/add_1Add siamese_4/scala4/batchnorm/mul_1siamese_4/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_4/scala4/ReluRelu siamese_4/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_4/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_4/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala5/splitSplit siamese_4/scala5/split/split_dimsiamese_4/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
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
siamese_4/scala5/Conv2DConv2Dsiamese_4/scala5/splitsiamese_4/scala5/split_1*
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
siamese_4/scala5/Conv2D_1Conv2Dsiamese_4/scala5/split:1siamese_4/scala5/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_4/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala5/concatConcatV2siamese_4/scala5/Conv2Dsiamese_4/scala5/Conv2D_1siamese_4/scala5/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_4/scala5/AddAddsiamese_4/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
O
score_2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
score_2/Conv2DConv2Dscore_2/splitConst_1*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
score_2/Conv2D_1Conv2Dscore_2/split:1Const_1*
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
score_2/Conv2D_2Conv2Dscore_2/split:2Const_1*
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
U
score_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_2/concatConcatV2score_2/Conv2Dscore_2/Conv2D_1score_2/Conv2D_2score_2/concat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
�
adjust_2/Conv2DConv2Dscore_2/concatadjust/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations

i
adjust_2/AddAddadjust_2/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:"3��C