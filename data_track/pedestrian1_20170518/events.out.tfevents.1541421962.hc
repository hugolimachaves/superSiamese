       �K"	  ����Abrain.Event:2��
�SP     ����	�����A"Ơ!
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_1Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
n
Placeholder_2Placeholder*
shape:*
dtype0*&
_output_shapes
:
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
is_training_1Const*
dtype0
*
_output_shapes
: *
value	B
 Z 
�
>siamese/scala1/conv/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@siamese/scala1/conv/weights*%
valueB"         `   *
dtype0
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
Hsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala1/conv/weights/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
seed2	*
dtype0*&
_output_shapes
:`*

seed
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
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala1/conv/weights*
	container *
shape:`*
dtype0*&
_output_shapes
:`
�
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`
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
=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala1/conv/weights/read*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: *
T0
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
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(
�
siamese/scala1/conv/biases/readIdentitysiamese/scala1/conv/biases*
_output_shapes
:`*
T0*-
_class#
!loc:@siamese/scala1/conv/biases
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
VariableV2*
shape:`*
dtype0*
_output_shapes
:`*
shared_name *)
_class
loc:@siamese/scala1/bn/beta*
	container 
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
VariableV2*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma*
	container *
shape:`*
dtype0*
_output_shapes
:`
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
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`*
dtype0*
_output_shapes
:`
�
$siamese/scala1/bn/moving_mean/AssignAssignsiamese/scala1/bn/moving_mean/siamese/scala1/bn/moving_mean/Initializer/Const*
_output_shapes
:`*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(
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
(siamese/scala1/bn/moving_variance/AssignAssign!siamese/scala1/bn/moving_variance3siamese/scala1/bn/moving_variance/Initializer/Const*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(
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
siamese/scala1/moments/meanMeansiamese/scala1/Add-siamese/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
#siamese/scala1/moments/StopGradientStopGradientsiamese/scala1/moments/mean*
T0*&
_output_shapes
:`
�
(siamese/scala1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1/Add#siamese/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
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
$siamese/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`*
dtype0*
_output_shapes
:`
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
8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biased*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Isiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *    
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
>siamese/scala1/siamese/scala1/bn/moving_mean/local_step/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepIsiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zeros*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(
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
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMul@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub$siamese/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
isiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biased@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Asiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x$siamese/scala1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
&siamese/scala1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0
�
7siamese/scala1/siamese/scala1/bn/moving_variance/biased
VariableV2*
shape:`*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container 
�
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
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
@siamese/scala1/siamese/scala1/bn/moving_variance/local_step/readIdentity;siamese/scala1/siamese/scala1/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read siamese/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub&siamese/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
ssiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Lsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepRsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x&siamese/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
 siamese/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
e
siamese/scala1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala1/cond/switch_tIdentitysiamese/scala1/cond/Switch:1*
_output_shapes
: *
T0

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
siamese/scala1/cond/Switch_1Switchsiamese/scala1/moments/Squeezesiamese/scala1/cond/pred_id*1
_class'
%#loc:@siamese/scala1/moments/Squeeze* 
_output_shapes
:`:`*
T0
�
siamese/scala1/cond/Switch_2Switch siamese/scala1/moments/Squeeze_1siamese/scala1/cond/pred_id*3
_class)
'%loc:@siamese/scala1/moments/Squeeze_1* 
_output_shapes
:`:`*
T0
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
N*
_output_shapes

:`: *
T0
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
siamese/scala1/batchnorm/addAddsiamese/scala1/cond/Merge_1siamese/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
j
siamese/scala1/batchnorm/RsqrtRsqrtsiamese/scala1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/mulMulsiamese/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/mul_1Mulsiamese/scala1/Addsiamese/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
siamese/scala1/batchnorm/mul_2Mulsiamese/scala1/cond/Mergesiamese/scala1/batchnorm/mul*
_output_shapes
:`*
T0
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
siamese/scala1/ReluRelusiamese/scala1/batchnorm/add_1*
T0*&
_output_shapes
:;;`
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
>siamese/scala2/conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@siamese/scala2/conv/weights*%
valueB"      0      
�
=siamese/scala2/conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *    
�
?siamese/scala2/conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *���<
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
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*'
_output_shapes
:0�*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
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
6siamese/scala2/conv/weights/Regularizer/l2_regularizerMul<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
,siamese/scala2/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala2/conv/biases*
valueB�*���=
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
siamese/scala2/conv/biases/readIdentitysiamese/scala2/conv/biases*-
_class#
!loc:@siamese/scala2/conv/biases*
_output_shapes	
:�*
T0
V
siamese/scala2/ConstConst*
_output_shapes
: *
value	B :*
dtype0
`
siamese/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/splitSplitsiamese/scala2/split/split_dimsiamese/scala1/poll/MaxPool*8
_output_shapes&
$:0:0*
	num_split*
T0
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
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
\
siamese/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
VariableV2*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala2/bn/beta*
	container *
shape:�*
dtype0
�
siamese/scala2/bn/beta/AssignAssignsiamese/scala2/bn/beta(siamese/scala2/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
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
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma**
_class 
loc:@siamese/scala2/bn/gamma*
_output_shapes	
:�*
T0
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
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
(siamese/scala2/bn/moving_variance/AssignAssign!siamese/scala2/bn/moving_variance3siamese/scala2/bn/moving_variance/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(
�
&siamese/scala2/bn/moving_variance/readIdentity!siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
siamese/scala2/moments/varianceMean(siamese/scala2/moments/SquaredDifference1siamese/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
siamese/scala2/moments/SqueezeSqueezesiamese/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
 siamese/scala2/moments/Squeeze_1Squeezesiamese/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container 
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
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
�
>siamese/scala2/siamese/scala2/bn/moving_mean/local_step/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepIsiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
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
isiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biased@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Asiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
siamese/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    
�
7siamese/scala2/siamese/scala2/bn/moving_variance/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape:�
�
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
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
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivGsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
e
siamese/scala2/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
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
siamese/scala2/batchnorm/mulMulsiamese/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala2/batchnorm/mul_1Mulsiamese/scala2/Addsiamese/scala2/batchnorm/mul*'
_output_shapes
:�*
T0
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
siamese/scala2/poll/MaxPoolMaxPoolsiamese/scala2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
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
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
siamese/scala3/conv/weights
VariableV2*.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
"siamese/scala3/conv/weights/AssignAssignsiamese/scala3/conv/weights8siamese/scala3/conv/weights/Initializer/truncated_normal*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
 siamese/scala3/conv/weights/readIdentitysiamese/scala3/conv/weights*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *o:
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
,siamese/scala3/conv/biases/Initializer/ConstConst*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala3/conv/biases*
valueB�*���=*
dtype0
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
siamese/scala3/Conv2DConv2Dsiamese/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�
�
siamese/scala3/AddAddsiamese/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
�
(siamese/scala3/bn/beta/Initializer/ConstConst*
_output_shapes	
:�*)
_class
loc:@siamese/scala3/bn/beta*
valueB�*    *
dtype0
�
siamese/scala3/bn/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala3/bn/beta*
	container *
shape:�
�
siamese/scala3/bn/beta/AssignAssignsiamese/scala3/bn/beta(siamese/scala3/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�
�
siamese/scala3/bn/beta/readIdentitysiamese/scala3/bn/beta*
T0*)
_class
loc:@siamese/scala3/bn/beta*
_output_shapes	
:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�
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
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala3/moments/meanMeansiamese/scala3/Add-siamese/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
#siamese/scala3/moments/StopGradientStopGradientsiamese/scala3/moments/mean*'
_output_shapes
:�*
T0
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
 siamese/scala3/moments/Squeeze_1Squeezesiamese/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
$siamese/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
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
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readsiamese/scala3/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/x@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
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
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
&siamese/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/AssignAssign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepMsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zeros*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub&siamese/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Rsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivGsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
e
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

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
siamese/scala3/cond/Switch_1Switchsiamese/scala3/moments/Squeezesiamese/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*1
_class'
%#loc:@siamese/scala3/moments/Squeeze
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
siamese/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala3/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
siamese/scala3/batchnorm/subSubsiamese/scala3/bn/beta/readsiamese/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/add_1Addsiamese/scala3/batchnorm/mul_1siamese/scala3/batchnorm/sub*
T0*'
_output_shapes
:

�
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
?siamese/scala4/conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *���<
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
8siamese/scala4/conv/weights/Initializer/truncated_normalAdd<siamese/scala4/conv/weights/Initializer/truncated_normal/mul=siamese/scala4/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
siamese/scala4/conv/weights
VariableV2*.
_class$
" loc:@siamese/scala4/conv/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
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
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *o:*
dtype0
�
=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala4/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: 
�
6siamese/scala4/conv/weights/Regularizer/l2_regularizerMul<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: 
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
siamese/scala4/ConstConst*
_output_shapes
: *
value	B :*
dtype0
`
siamese/scala4/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/split_1Split siamese/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala4/Conv2DConv2Dsiamese/scala4/splitsiamese/scala4/split_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
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
(siamese/scala4/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala4/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/beta
VariableV2*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala4/bn/beta*
	container *
shape:�*
dtype0
�
siamese/scala4/bn/beta/AssignAssignsiamese/scala4/bn/beta(siamese/scala4/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�
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
VariableV2**
_class 
loc:@siamese/scala4/bn/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
siamese/scala4/bn/gamma/readIdentitysiamese/scala4/bn/gamma*
_output_shapes	
:�*
T0**
_class 
loc:@siamese/scala4/bn/gamma
�
/siamese/scala4/bn/moving_mean/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    
�
siamese/scala4/bn/moving_mean
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
(siamese/scala4/bn/moving_variance/AssignAssign!siamese/scala4/bn/moving_variance3siamese/scala4/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
1siamese/scala4/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
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
 siamese/scala4/moments/Squeeze_1Squeezesiamese/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    
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
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container 
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
<siamese/scala4/siamese/scala4/bn/moving_mean/local_step/readIdentity7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
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
Lsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepLsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/x@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container 
�
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala4/siamese/scala4/bn/moving_variance/biased/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biased*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Msiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *    
�
;siamese/scala4/siamese/scala4/bn/moving_variance/local_step
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 
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
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
siamese/scala4/cond/switch_fIdentitysiamese/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala4/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
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
siamese/scala4/batchnorm/addAddsiamese/scala4/cond/Merge_1siamese/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
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
siamese/scala4/batchnorm/mul_1Mulsiamese/scala4/Addsiamese/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
siamese/scala4/batchnorm/mul_2Mulsiamese/scala4/cond/Mergesiamese/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/subSubsiamese/scala4/bn/beta/readsiamese/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
=siamese/scala5/conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *    
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
<siamese/scala5/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala5/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
8siamese/scala5/conv/weights/Initializer/truncated_normalAdd<siamese/scala5/conv/weights/Initializer/truncated_normal/mul=siamese/scala5/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
siamese/scala5/conv/weights
VariableV2*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights
�
"siamese/scala5/conv/weights/AssignAssignsiamese/scala5/conv/weights8siamese/scala5/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��
�
 siamese/scala5/conv/weights/readIdentitysiamese/scala5/conv/weights*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *o:
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
_output_shapes	
:�
V
siamese/scala5/ConstConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese/scala5/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
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
siamese/scala5/Conv2DConv2Dsiamese/scala5/splitsiamese/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
siamese/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
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
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`*
	dilations
*
T0
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
!siamese/scala1_1/moments/varianceMean*siamese/scala1_1/moments/SquaredDifference3siamese/scala1_1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
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
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Nsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese/scala1_1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese/scala1_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
g
siamese/scala1_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala1_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala1_1/cond/Switch_1Switch siamese/scala1_1/moments/Squeezesiamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese/scala1_1/moments/Squeeze
�
siamese/scala1_1/cond/Switch_2Switch"siamese/scala1_1/moments/Squeeze_1siamese/scala1_1/cond/pred_id*5
_class+
)'loc:@siamese/scala1_1/moments/Squeeze_1* 
_output_shapes
:`:`*
T0
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
siamese/scala1_1/cond/MergeMergesiamese/scala1_1/cond/Switch_3 siamese/scala1_1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese/scala1_1/cond/Merge_1Mergesiamese/scala1_1/cond/Switch_4 siamese/scala1_1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese/scala1_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala1_1/batchnorm/addAddsiamese/scala1_1/cond/Merge_1 siamese/scala1_1/batchnorm/add/y*
_output_shapes
:`*
T0
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
 siamese/scala1_1/batchnorm/mul_2Mulsiamese/scala1_1/cond/Mergesiamese/scala1_1/batchnorm/mul*
_output_shapes
:`*
T0
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
siamese/scala1_1/ReluRelu siamese/scala1_1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese/scala1_1/poll/MaxPoolMaxPoolsiamese/scala1_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:==`
X
siamese/scala2_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
"siamese/scala2_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/split_1Split"siamese/scala2_1/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2_1/Conv2DConv2Dsiamese/scala2_1/splitsiamese/scala2_1/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
�
siamese/scala2_1/Conv2D_1Conv2Dsiamese/scala2_1/split:1siamese/scala2_1/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
^
siamese/scala2_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/concatConcatV2siamese/scala2_1/Conv2Dsiamese/scala2_1/Conv2D_1siamese/scala2_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:99�
�
siamese/scala2_1/AddAddsiamese/scala2_1/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese/scala2_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2_1/moments/meanMeansiamese/scala2_1/Add/siamese/scala2_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
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
!siamese/scala2_1/moments/varianceMean*siamese/scala2_1/moments/SquaredDifference3siamese/scala2_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese/scala2_1/moments/SqueezeSqueezesiamese/scala2_1/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese/scala2_1/moments/Squeeze_1Squeeze!siamese/scala2_1/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese/scala2_1/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Nsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
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
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0
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
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze
�
siamese/scala2_1/cond/Switch_2Switch"siamese/scala2_1/moments/Squeeze_1siamese/scala2_1/cond/pred_id*5
_class+
)'loc:@siamese/scala2_1/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
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
siamese/scala2_1/cond/MergeMergesiamese/scala2_1/cond/Switch_3 siamese/scala2_1/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese/scala2_1/cond/Merge_1Mergesiamese/scala2_1/cond/Switch_4 siamese/scala2_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala2_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/batchnorm/addAddsiamese/scala2_1/cond/Merge_1 siamese/scala2_1/batchnorm/add/y*
T0*
_output_shapes	
:�
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
 siamese/scala2_1/batchnorm/mul_1Mulsiamese/scala2_1/Addsiamese/scala2_1/batchnorm/mul*'
_output_shapes
:99�*
T0
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
siamese/scala2_1/poll/MaxPoolMaxPoolsiamese/scala2_1/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
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
siamese/scala3_1/moments/meanMeansiamese/scala3_1/Add/siamese/scala3_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
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
!siamese/scala3_1/moments/varianceMean*siamese/scala3_1/moments/SquaredDifference3siamese/scala3_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese/scala3_1/moments/SqueezeSqueezesiamese/scala3_1/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese/scala3_1/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Csiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese/scala3_1/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala3_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0
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
usiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Nsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese/scala3_1/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
g
siamese/scala3_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

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
siamese/scala3_1/cond/Switch_2Switch"siamese/scala3_1/moments/Squeeze_1siamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese/scala3_1/moments/Squeeze_1
�
siamese/scala3_1/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
 siamese/scala3_1/batchnorm/RsqrtRsqrtsiamese/scala3_1/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese/scala4_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala4_1/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala4_1/split_1Split"siamese/scala4_1/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala4_1/Conv2DConv2Dsiamese/scala4_1/splitsiamese/scala4_1/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
^
siamese/scala4_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
!siamese/scala4_1/moments/varianceMean*siamese/scala4_1/moments/SquaredDifference3siamese/scala4_1/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala4_1/moments/SqueezeSqueezesiamese/scala4_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese/scala4_1/moments/Squeeze_1Squeeze!siamese/scala4_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese/scala4_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese/scala4_1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese/scala4_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese/scala4_1/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese/scala4_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
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
Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
siamese/scala4_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala4_1/cond/Switch_1Switch siamese/scala4_1/moments/Squeezesiamese/scala4_1/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese/scala4_1/moments/Squeeze
�
siamese/scala4_1/cond/Switch_2Switch"siamese/scala4_1/moments/Squeeze_1siamese/scala4_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala4_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4_1/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese/scala4_1/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
siamese/scala4_1/batchnorm/mulMul siamese/scala4_1/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/batchnorm/mul_1Mulsiamese/scala4_1/Addsiamese/scala4_1/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese/scala4_1/batchnorm/mul_2Mulsiamese/scala4_1/cond/Mergesiamese/scala4_1/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala4_1/batchnorm/subSubsiamese/scala4/bn/beta/read siamese/scala4_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/batchnorm/add_1Add siamese/scala4_1/batchnorm/mul_1siamese/scala4_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala4_1/ReluRelu siamese/scala4_1/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese/scala5_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
"siamese/scala5_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/split_1Split"siamese/scala5_1/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*
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
�
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese/scala5_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/concatConcatV2siamese/scala5_1/Conv2Dsiamese/scala5_1/Conv2D_1siamese/scala5_1/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese/scala5_1/AddAddsiamese/scala5_1/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
m
score/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
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
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*
T0*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split
�
score/Conv2DConv2Dscore/split_1score/split*
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
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
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
score/Conv2D_3Conv2Dscore/split_1:3score/split:3*&
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
score/Conv2D_4Conv2Dscore/split_1:4score/split:4*
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
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
S
score/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
score/concatConcatV2score/Conv2Dscore/Conv2D_1score/Conv2D_2score/Conv2D_3score/Conv2D_4score/Conv2D_5score/Conv2D_6score/Conv2D_7score/concat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
o
score/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose_1	Transposescore/concatscore/transpose_1/perm*
T0*&
_output_shapes
:*
Tperm0
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
VariableV2*!
_class
loc:@adjust/weights*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
adjust/weights/AssignAssignadjust/weights adjust/weights/Initializer/Const*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(
�
adjust/weights/readIdentityadjust/weights*!
_class
loc:@adjust/weights*&
_output_shapes
:*
T0
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@adjust/biases*
	container 
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
adjust/biases/readIdentityadjust/biases* 
_class
loc:@adjust/biases*
_output_shapes
:*
T0
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
/adjust/biases/Regularizer/l2_regularizer/L2LossL2Lossadjust/biases/read*
T0* 
_class
loc:@adjust/biases*
_output_shapes
: 
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

adjust/AddAddadjust/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
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
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
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
save/Assign_4Assignsiamese/scala1/bn/moving_meansave/RestoreV2:4*
_output_shapes
:`*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(
�
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2:5*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save/Assign_6Assignsiamese/scala1/conv/biasessave/RestoreV2:6*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases
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
save/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave/RestoreV2:8*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
save/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave/RestoreV2:10*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave/RestoreV2:11*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/Assign_12Assignsiamese/scala2/bn/betasave/RestoreV2:12*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(
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
save/Assign_14Assignsiamese/scala2/bn/moving_meansave/RestoreV2:14*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2:17*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
save/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave/RestoreV2:18*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave/RestoreV2:19*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
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
save/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave/RestoreV2:21*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2:23*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2:24*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
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
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2:26*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases
�
save/Assign_27Assignsiamese/scala3/conv/weightssave/RestoreV2:27*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��
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
save/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave/RestoreV2:29*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2:30*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2:31*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: 
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
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2:34*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
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
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
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
save/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave/RestoreV2:40*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(
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
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2:42*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_43Assignsiamese/scala5/conv/weightssave/RestoreV2:43*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:;;`*
	dilations
*
T0*
strides
*
data_formatNHWC*
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
siamese_1/scala1/moments/meanMeansiamese_1/scala1/Add/siamese_1/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala1/moments/StopGradientStopGradientsiamese_1/scala1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese_1/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
�
3siamese_1/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_1/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_1/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_1/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_1/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
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
Nsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_1/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
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
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese_1/scala1/cond/Switch_1Switch siamese_1/scala1/moments/Squeezesiamese_1/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala1/moments/Squeeze* 
_output_shapes
:`:`
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
 siamese_1/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala1/batchnorm/addAddsiamese_1/scala1/cond/Merge_1 siamese_1/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_1/scala1/batchnorm/RsqrtRsqrtsiamese_1/scala1/batchnorm/add*
_output_shapes
:`*
T0
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
 siamese_1/scala1/batchnorm/mul_2Mulsiamese_1/scala1/cond/Mergesiamese_1/scala1/batchnorm/mul*
_output_shapes
:`*
T0
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
siamese_1/scala1/ReluRelu siamese_1/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*
ksize
*
paddingVALID*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides

X
siamese_1/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_1/scala2/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese_1/scala2/split_1Split"siamese_1/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_1/scala2/Conv2DConv2Dsiamese_1/scala2/splitsiamese_1/scala2/split_1*'
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
�
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
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
/siamese_1/scala2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_1/scala2/moments/meanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/mean*'
_output_shapes
:�*
T0
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
 siamese_1/scala2/moments/SqueezeSqueezesiamese_1/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala2/moments/Squeeze_1Squeeze!siamese_1/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
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
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_1/scala2/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_1/scala2/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
siamese_1/scala2/cond/switch_fIdentitysiamese_1/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala2/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala2/cond/Switch_1Switch siamese_1/scala2/moments/Squeezesiamese_1/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_1/scala2/moments/Squeeze
�
siamese_1/scala2/cond/Switch_2Switch"siamese_1/scala2/moments/Squeeze_1siamese_1/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_1/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese_1/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_1/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/MergeMergesiamese_1/scala2/cond/Switch_3 siamese_1/scala2/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_1/scala2/cond/Merge_1Mergesiamese_1/scala2/cond/Switch_4 siamese_1/scala2/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
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
siamese_1/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_1/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala2/batchnorm/add_1Add siamese_1/scala2/batchnorm/mul_1siamese_1/scala2/batchnorm/sub*'
_output_shapes
:�*
T0
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
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
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
!siamese_1/scala3/moments/varianceMean*siamese_1/scala3/moments/SquaredDifference3siamese_1/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala3/moments/Squeeze_1Squeeze!siamese_1/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_1/scala3/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_1/scala3/moments/Squeeze*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_1/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_1/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
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
 siamese_1/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_1/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_1/scala3/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
usiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_1/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
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
siamese_1/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

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
siamese_1/scala3/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala3/cond/Switch_1Switch siamese_1/scala3/moments/Squeezesiamese_1/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_2Switch"siamese_1/scala3/moments/Squeeze_1siamese_1/scala3/cond/pred_id*5
_class+
)'loc:@siamese_1/scala3/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
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
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala3/cond/Merge_1Mergesiamese_1/scala3/cond/Switch_4 siamese_1/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
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
 siamese_1/scala3/batchnorm/mul_1Mulsiamese_1/scala3/Addsiamese_1/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
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
siamese_1/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_1/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_1/scala4/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_1/scala4/split_1Split"siamese_1/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_1/scala4/Conv2DConv2Dsiamese_1/scala4/splitsiamese_1/scala4/split_1*
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
�
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*
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
^
siamese_1/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/concatConcatV2siamese_1/scala4/Conv2Dsiamese_1/scala4/Conv2D_1siamese_1/scala4/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
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
siamese_1/scala4/moments/meanMeansiamese_1/scala4/Add/siamese_1/scala4/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
!siamese_1/scala4/moments/varianceMean*siamese_1/scala4/moments/SquaredDifference3siamese_1/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
&siamese_1/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
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
Nsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Isiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
c
siamese_1/scala4/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala4/cond/switch_tIdentitysiamese_1/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala4/cond/switch_fIdentitysiamese_1/scala4/cond/Switch*
_output_shapes
: *
T0

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
siamese_1/scala4/cond/Switch_2Switch"siamese_1/scala4/moments/Squeeze_1siamese_1/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_1/scala4/moments/Squeeze_1
�
siamese_1/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
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
siamese_1/scala4/batchnorm/mulMul siamese_1/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_1Mulsiamese_1/scala4/Addsiamese_1/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
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
 siamese_1/scala5/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_1/scala5/splitSplit siamese_1/scala5/split/split_dimsiamese_1/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_1/scala5/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
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
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala5/Conv2D_1Conv2Dsiamese_1/scala5/split:1siamese_1/scala5/split_1:1*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

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
Placeholder_4Placeholder*&
_output_shapes
:*
shape:*
dtype0
r
Placeholder_5Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
n
Placeholder_6Placeholder*
shape:*
dtype0*&
_output_shapes
:
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
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:,*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save_1/Assign_1Assignadjust/weightssave_1/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_2Assignsiamese/scala1/bn/betasave_1/RestoreV2:2*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(
�
save_1/Assign_3Assignsiamese/scala1/bn/gammasave_1/RestoreV2:3*
_output_shapes
:`*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(
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
save_1/Assign_5Assign!siamese/scala1/bn/moving_variancesave_1/RestoreV2:5*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
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
save_1/Assign_7Assignsiamese/scala1/conv/weightssave_1/RestoreV2:7*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(
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
save_1/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave_1/RestoreV2:11*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
save_1/Assign_12Assignsiamese/scala2/bn/betasave_1/RestoreV2:12*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_15Assign!siamese/scala2/bn/moving_variancesave_1/RestoreV2:15*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
save_1/Assign_16Assignsiamese/scala2/conv/biasessave_1/RestoreV2:16*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(
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
save_1/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave_1/RestoreV2:19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
save_1/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave_1/RestoreV2:21*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_22Assignsiamese/scala3/bn/betasave_1/RestoreV2:22*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(
�
save_1/Assign_23Assignsiamese/scala3/bn/gammasave_1/RestoreV2:23*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(
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
save_1/Assign_25Assign!siamese/scala3/bn/moving_variancesave_1/RestoreV2:25*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save_1/Assign_27Assignsiamese/scala3/conv/weightssave_1/RestoreV2:27*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(
�
save_1/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave_1/RestoreV2:28*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave_1/RestoreV2:29*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave_1/RestoreV2:30*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
save_1/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave_1/RestoreV2:31*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
save_1/Assign_32Assignsiamese/scala4/bn/betasave_1/RestoreV2:32*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_36Assignsiamese/scala4/conv/biasessave_1/RestoreV2:36*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave_1/RestoreV2:39*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave_1/RestoreV2:40*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
save_1/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave_1/RestoreV2:41*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
save_1/Assign_43Assignsiamese/scala5/conv/weightssave_1/RestoreV2:43*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
�
siamese_2/scala1/Conv2DConv2DPlaceholder_4 siamese/scala1/conv/weights/read*
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
siamese_2/scala1/AddAddsiamese_2/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
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
3siamese_2/scala1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_2/scala1/moments/varianceMean*siamese_2/scala1/moments/SquaredDifference3siamese_2/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_2/scala1/moments/SqueezeSqueezesiamese_2/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
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
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_2/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Nsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
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
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
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
Tsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese_2/scala1/cond/Switch_2Switch"siamese_2/scala1/moments/Squeeze_1siamese_2/scala1/cond/pred_id*5
_class+
)'loc:@siamese_2/scala1/moments/Squeeze_1* 
_output_shapes
:`:`*
T0
�
siamese_2/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_2/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
�
siamese_2/scala1/cond/MergeMergesiamese_2/scala1/cond/Switch_3 siamese_2/scala1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
�
siamese_2/scala1/cond/Merge_1Mergesiamese_2/scala1/cond/Switch_4 siamese_2/scala1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
e
 siamese_2/scala1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_2/scala1/batchnorm/addAddsiamese_2/scala1/cond/Merge_1 siamese_2/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
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
siamese_2/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_2/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
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
 siamese_2/scala2/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
"siamese_2/scala2/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala2/split_1Split"siamese_2/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*
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
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*
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
^
siamese_2/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
siamese_2/scala2/moments/meanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_2/scala2/moments/StopGradientStopGradientsiamese_2/scala2/moments/mean*'
_output_shapes
:�*
T0
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
 siamese_2/scala2/moments/SqueezeSqueezesiamese_2/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_2/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Nsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_2/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
(siamese_2/scala2/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9
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
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_2/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese_2/scala2/cond/Switch_2Switch"siamese_2/scala2/moments/Squeeze_1siamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala2/moments/Squeeze_1
�
siamese_2/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_2/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_2/scala2/cond/Merge_1Mergesiamese_2/scala2/cond/Switch_4 siamese_2/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala2/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
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
siamese_2/scala2/batchnorm/mulMul siamese_2/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
/siamese_2/scala3/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_2/scala3/moments/meanMeansiamese_2/scala3/Add/siamese_2/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
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
"siamese_2/scala3/moments/Squeeze_1Squeeze!siamese_2/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_2/scala3/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0
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
ksiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_2/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
(siamese_2/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_2/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese_2/scala3/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

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
siamese_2/scala3/cond/pred_idIdentityis_training_2*
T0
*
_output_shapes
: 
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
siamese_2/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese_2/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
 siamese_2/scala3/batchnorm/RsqrtRsqrtsiamese_2/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/mulMul siamese_2/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_2/scala3/batchnorm/mul_1Mulsiamese_2/scala3/Addsiamese_2/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
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
 siamese_2/scala4/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_2/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/split_1Split"siamese_2/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*
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
siamese_2/scala4/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_2/scala4/AddAddsiamese_2/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_2/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
!siamese_2/scala4/moments/varianceMean*siamese_2/scala4/moments/SquaredDifference3siamese_2/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_2/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_2/scala4/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
"siamese_2/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
g
siamese_2/scala4/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

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
siamese_2/scala4/cond/Switch_1Switch siamese_2/scala4/moments/Squeezesiamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_2/scala4/moments/Squeeze
�
siamese_2/scala4/cond/Switch_2Switch"siamese_2/scala4/moments/Squeeze_1siamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala4/moments/Squeeze_1
�
siamese_2/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_2/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
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
N*
_output_shapes
	:�: *
T0
e
 siamese_2/scala4/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_2/scala4/batchnorm/addAddsiamese_2/scala4/cond/Merge_1 siamese_2/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala4/batchnorm/RsqrtRsqrtsiamese_2/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala4/batchnorm/mulMul siamese_2/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_2/scala5/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_2/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/splitSplit siamese_2/scala5/split/split_dimsiamese_2/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
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
siamese_2/scala5/Conv2DConv2Dsiamese_2/scala5/splitsiamese_2/scala5/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*
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
^
siamese_2/scala5/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
value��B���"���ޛ���=��l<J���@F�:�Ɇ=���<�en���]=$6��4��<��=9�����	�"eM=�����=���[= ѵ����eS�����<��׻��<=8Pʼ�5&��RK��IG=��<�3p��ۼ�������q<���<@�>|��=�VȽ=�F8�P�׽��׼dU~�6e$=�|O��^���/���j;~t� ?���<��X��E׽FF4=��h<L�D����:��>�8���]�o����>��<h'�<��<s�=��<�j̼ ����������ٽxmD<�9��m>X�=O�� ���ѕ��2=�"˴�w=���D=PB��t��<��ս��<��i�z���\_s��N�<�����L׽�\��*��Ǆ�܇�&���
<����6�:�Q����(�<VK[�
�=���=��= �<���l<�#���'��@�g;���=�����=�ڽ���<P<E���Ԃ��b���r���t=�˼�,=(�����<lb���p=)r=��Ԥ˼@,|�N�E���<��:����^
H�໡�L��� �<�/<�v�<�>��`}��&�=�&f�D)�<@=��yp<h<��F<H��<��?;l�ټp���KC<�5<d��\��<P%:<��J<�Ի��Q�xA��L�<D�����<H�伌�&=�)�<�V�<��ݼ(�f�`�<���<����G����<p���ȕ���`<�7�<�3�<>i�^ʇ� �����[�<t�f=�T�=�S=�P[;�0��(w+<�R;�}�<`7�`D�;�t�<��y=�i���=�M���H<�`Y; �b<δ=�Ң�����}#=tdp=X<p�����<�}W���<@�[=dώ<pю�,5Ǽ(�=�(���<pp����G<,ށ<Ц��p#�;bmV=0W2��5�<�K
��`�� ��✻��J�p��;lZ�<�P˻��ʻ�����yż젴<���;��I=�Z.=�#=l���P#�;@t
��;=�\���*��2�=�n=�.�$1��<!F����=|��<���w��=0��(d<�j��B���e?�a�_=�%�Ǖ�=`�k;�7�=(Uh<���:߼@3m�`��;���<�n�;d���P�;��)<��C=l��6=�jR=�����I=�M>��=D���"��):Nmܽ�U=��l�?�<����C��ħ��*%�U -�8:�<L*>�⌻���/=4���lne������= 	�����<j��*>$Iɼ���(胼ir�=�)�=dU��8d���ü o�9����$w����,��>}��=�.�P,��0�.��=���y(��
7�|H�<4>��`�ܼ�ݽp5�}���Ljս
��h贼�jI��걽�h�<�n2<�˲�+_���J���M�;��� =����4?U� �#;���<U�u=tw�<U=� <&3L������ ��h���E=�6>�{���>9ѽ^	=�ʼ�{{�=�8�n_�tP���=��< ����`J�d��<������s=[�= �*����H%
<�w� �3�İ��r��'�@Kջ����x�N<8�;��X�<~��1�U��=:��
M=��`�;p��;���<�M=8�<`b˼���8x<9_=|Ή�E�1=�˶<h�'=�ڳ��C���%ڻ\��<֡�Ȁ=����d=�K=�.�<P<�"У� �9؍+<Fo
��Yd�X�<��J�����<�=�=6�g�.���ḽ�V�;� �<>n=%׷=�%q=@�;٤����<P�Ի�.;=0�l�8,?���A=P��=�OA�t�q= �G�0��;(P����<D�=�
";@�-��j=�2E=��i;�uv�lp�<D���@}�;�`|=�\�< �������_=h[���h=Ƞ� �Y:���<��)� xh�_;�=� k�p��<�6�8�>�$����v�����^m=�@B=����8B�:l%�T�M��a�<P�G<�os=�j=��=�8޻`�9<�����9=�}X��|m�1G�=s!�=���ؔݼ����?= ,d��w�z>�5�=؊�<Y��=舽 �+��"W<.WI��ҍ<H�=_�=�a=����5<�zּ`�c��s;<��!=H�<����ｼ�=k�ｉ�>��=P�ռw�=�3�=O�y=� ��o��US�=?���=�{�� �R����<�w������t��\n�Zč�0�=pK��\ɞ�("0=�u�� ��;86����лx���V�=\���E�=T����ý �a=К�<���<X��<�i=�=��v<�<�<�jZ���6����=�8=������	�I�<����i�<����>��p�<b������r���mw��
u½L����E��T���M���H�=�Q=�#Q����0�m�px�;��&<h��<��̽Ig�= ۿ;�&�=�}�<Ƌ����ʺ��G=�c�<��F��:=@��;;�e=��=�X�=���=ܓּ
�=6eR���� �ZN
=L�F�h��=uh=� p���Ƽ��<=���5=��=(�0� x7��<��� ��: ,��s��ر}�@��<��<i�<@��;Pd��l�\�ϼ>�=�����S%=�ݼ@�I�`i)<���=[�=a�=�> ;��0�J�=��%=pNӻ�D�<��L<�f=�ط;J�<`k;���<|V�� !=��;��0=P ^=�o�<��_��L�@����;��� f��8	=h�6���X�1<�
=M\=`٫��"��=m��9k<H�<�Ȓ=*W=�Cd=$�<�I�<��<H���F)=�;�����:=��=t]��h��<|��<F�=�<��;P�< '�;�$9<|4�=�;Q<��;�VV;@�d< Y-������,=$y�< -(� L���H=h���;<@!�`%����; ̤���;VĿ=(� ��֨<�F:�H漼P]��R�`e��f�=?i=������</���bA��(�<�)�<��<d��<�g =�-<��R<����y==Аٻ�Q�9��=R&�=ZF�����z��妼@�c��3p�5��=�3�=���<��ٽF�}�5�=��g��}8������ɭ=��=�?*=DtU�H��<0Ό<HNἀY<p(�<�+=X�ܼ����d��<�7����>G��=����+�=��	�%P&=ԫ3��w�<f�=0q4<r��=T�q�`�z���<
�9�~�^����;L$+��[�=��.> �M��3=%)0=x��<w��=��N���+�،r��2�=�0����~=�+�ز���>�������8=�4�<\��=�%�<|"�=���i�<{��=�i�;f�;�p�x.�<H�hR�<�e��FEs�4��<Pʠ���=��Ȼ�g�P�O��������<"�	��DW=:�W=%�E=��R<�����s<�MA=%\F=PW_�����>P1�<�=�4����w�x��K�S=1�=j��Xmq=ɹb=�F=n�=ۑ�=�C�=�. =2=���Ĩ���R�"S=L�	��==��=���X�ۼx3�<�o>�f==:f==h�.�����=Ӽ�!<;�Ү;����Dg����<��C=��<@�c;�ӱ<�~!<����{�= i۹��W=�݀<8=�<�l<��=!m=*=�x�<�"»��<�>W=�/�:u<!=�=..=X]Y<���<`f�;�"<0���2=��%<��=��V=S�V= G; �张i�G�<�	S����:�v=�i��H�~���`=��_=�U�<T�� Ӧ�������<���<j��=C�f=��=�z<�:����<d�<@��=$��<�z��v =$/�=�����<�&U=�~=x8n<�4�<�E=4g�<�\<�q=8��<x��<�1=p®;l�������TB�<��=��~<���<H�v=��;�d�<#�o=X<���<@���'=���=�����;�i�:�ꎼ�{�@����r��4ѫ=���=�lZ<`�O<�Ѝ�����C/3=x��<��=	�'=^�=�h�<�=Y=��<��=2�<��
�p��;0�M=8���ϼ,�������0'輰]�;C��=��=�=V������[=����H����;%ܗ=re=�ӭ��H��*�=2ǀ=�Y� ��<@��;0�C��,����<(�G<'8K=<ă=��=P����i=^P1� �!=P�ǻL��<kt�=�B<P��<�⁼ �=<����T���疧���< 8���=�>�'ܻ�f=���<KN=-I�=���<T��I�=H]D��t�< (������>������b����=�;p�=!�=*��=�����=Z�=(�<�>o=��F=��a<xO����g<�I޽�]���<�W��} �=�C�<���:��N�t��Y��=d�=��=�U;�P��;ϲ4=��p;��=�X�=<��=  s�P8o�h��=�q�=�=4v���ѥ;������>=W��=���;N�P=�3�=0�<���&= A�;B��=���< ����&�����<�>=B� �;]4=v�= �v�l2����=\ =$�<��j�'f����4�=�$O���$������ȼ�j��x�u�V6g=Dy�(A8���=x=�Ԧ�`\	>@����=2�=�/u=\4�<�f�;p-=0�;`~�<xmv=�s��{�=H��5u�=��=T��<�)+<��Ż�2=�^��V�4��<��
���<��==)��=�;t;��0�<k�<ȨE=�����m����;	%�z���->\r�=�A��/���E�<P���q�<��<t�=���=��=f�3��j�
=҉�=�r�=/�T=Ԩ�<@ʸ;�;p=�۽�pJ=trx=�k���U���)=t/	>t�(=ؖm�eJ=��v=�p=IĊ=�>��`���� �P�;O�'=@_�<�q=���=���<7�y=��2>t�I=b�=8k��:�=���=v³�|j��a�=Ѓ��${��ͼ����x>=�E*>\�t=�?ڻh6s��M����=(_<E&2>���=��<d��<v��=�< �>r�=�g��μ�q�<E�; ܺ��{��䟽^�½�d5=}�c=8g<U= {@;�ƼC�=���<��輎�'=��|==g-=y���H�B�<��<|j�=|B��z�L=P�z�����NQ�$7�< p�84ڀ=�j���=�;�Y�d=B�����<�k�<��0=��=��=�ʟ��y=���<6)-�0:E�oX��о�;P���Nu>�R�=�z�=�cc=�˃=R�(=���=4`��\�<�=�T�<���<p�`kC<d�=�0�=e�Ž��W��ȹ=���;�+�=:�k=�[2=V*Q�x�>�x<=�x�;��=���=��o<��$=�w8=������pd�;V�Y����=tm�<	�o=�d�<<Z��%�=�G����!>��}���:�R�=p=�\=��=H��=�@ý]�;=�S�=�X�=&�=`r���Gd=(��b=t@=0լ;�&�=��=ګ�/｀<f8���f�=���<��]:������=��=���<���<�ʕ=XU�$z˼G=p��<��p<`�t��j��r�����<��R� �v�|�� w��딽������-=��8��r9����='�,=]彦�>h7@�a��=?��=��A=��< ����' =�^�����<���=R""�lË=�A-����=,,�=�< c�; �u�Z�=P�ۻD1��L� =t ��쐘<��)=�c�= ���`(��<H,>=��s��߂���;�pC��)��-�>*.�=�3���r���h�<�y��
�<JM=bt=~k�=;+C=��_�"���G=?S�=
��=+��=���<�&�;Ÿ0=�X�TF=
B�=x�ܼ��X�i=��> ��<�=��1�=S��=ǂx=��=،s�� ���5)�0���6=��l<�Ga=�,�=8;k<�#c=LJ>��Y=���=x�,�?�=�En=�!��P�缻؜=@IJ��}���/�T ����=��>>	`�=���:�ƼYܻ�X�=\
�<'y5>���=��<�pL<xr�=pDp<��>��A<HC���{"��}�<`�Z;|7�0�û��>�i=Bإ��b�<@�޻P���.�J=2�X���X�`��;���ĩ�=D���<p�<��B�7���*t� ]�<H'-<@��<DWּ�O
���;��=�n6� �Q�z2�x��=c�C�=nƃ=�e �`]��G6��ķ���WQ� ����<t��VE!�8�<���N�����0�廪ǽ��X�M��8Y���>�TՃ<"�=Xe�rYL���ѽ��=�V\<8�&��_<[��=\w�<r�K��JZ� ��:>�R����L��<֯
��I�=�ݝ=�/� .n�@���(=�nת�����I=`�J���=yh��X0�<�9X����>��XPJ<�$���W��nU�X�E<o	������1��T�v��n�z�=���pｰ-h<*	��	|=��<�"�����<!	��W� =I�ŽP�� �=<{h�=X�ѽ'�=�om���5�ط\<�����5�����V�'=(� �䛽<,2� �ж���<=�<=��7<d����`�N��ҏ<T�< i��M��� ��q��0��;�l�< ��9Ҭ� �<�d�<�{�4�üT��`2;@��~�=��(:��U��$��jF��u~<��|���ڼ�3ú �@�F���_�� {�<$�|t�<��E��̈<$k��Э�;��%;�=�;����L��P�什3���ӌ;X$|�\�<��<���;�Dƻ��<�i=�Ya���%����L����<|�= h<<�=�<x4�<���`Qb��m
�؛9������
� s;BNr=@����ѻ�v��`�< ��9�ڼ��z����@е�T��<lݩ<��Ⱥ̕)� ��;#�9�<���<��:��j�N�hP��ټ0@��$	������0��(w<@��:nA=`��� �!<b.���Լ@j<`O;lc���3;���О'������`������@��;����|䇼@Q�;��=������D����<�����0���=��I=J�!���<���g�4�>DK�<��νK�s=0�j�v4�ؔ<<M�������H,= А:<�=��<��=�s�<ܳ�<d��<0_k��w6<@��;��1=�5���~Ѽ#Q��*�=_㺽�3=�'<,�<Љ<�5�=f �=�;齖����ao��\�ȍ��(�l��<:�U�V�q�TU�< OռjS�ǾY�,�ɼ��%���C��h�J᤽xb<:u=h9���m��a�><��H�W�(֞����=��n=����4��0tѼ0O\� H���Ѽ�����=;��=������l��X�tJ)��<<X�<����8ޭ�੽��1��u��D���(5
��.%�Lp�<|�� ��L�<G�ӽ-�޽J=���d�*�j���f=&���pXu��q
�8�c<��b=��`�a;�M=�����i�k���O=B+�=(�P��H�=�����pռ6��z�G�*/�h�!����e"=8�<s� �v�$��<D�ͼ��<o�P=x3ʼ>�����;��Q�@�k����^ʼ�9r�(*S����� ��9�幺�%�;���丼��=�'9�P�[<X/9����;�s&�`�<X�z<���<P�7�,�-�@в��M?<8�ȼ� �<�>;�t	<��ܼ �y�<�� 3-;4�鼠:�<�#��+�<, �<���<L�P�O�,��� �M�"���&��l��<`K�������qg<�?�<�
=.#l�`&I��s�0}~���;v�=`�Q=�3/=`W�;6*U�ؤ<���L#�< ,Ҽh���h�<�h=f� *�<�����;0&v��yS:�yt< ܻ��漋�&=�B|<�4�:�`���tA<@5μ�3c;�w$=��%<���|���0��<�Ѽ Nl: -l�X�X� �}; �����j;�aL=< $�p_=<@�:��bɼ(����T��I�4I�<`o�<ܖǼDu�ZQ�47��$�;H�Y�4R�<ڰ=��=�9����H�TF�T�=`�������Ȍ=�g=Nsz������ �T��=���<��Ľ��>h=����2Gy�v*=���ļ=���m�B=F�=�|=�hM= �7< �<>ф� H�9pO�;G�=���<��C�@u����=+h���>�==�J<L^�<��Z=
��=��������d�<|���,&=�����*���T<�Ņ�@�I�h���½�d�B��=$ͣ� ���fʼ����8����x��L쎼K��x�=����=��;�����6=�ɫ<O�=<�����<���<p�;pf�;V�.�0|&���=��R=p�����H_�`���,��<��� %���B�<|������r>��c����Ӹ�7#�>�g�P-׼��^�:�=��$=�2n��lB�0�������G=h�����=Xm����w=��=�E��PJ�����=Pj$�"���h;Œ��y]=D��=��`=9��=~c��{I���D��8�������<0!��=~h=�� ��<��ڻ0`)�0�<��B= v�;��k�<<&ἠb;`�q;�}�0����p�;���<�0�<��P<`�t��z��dݤ��Z=hn��<���@�� <l:b�=��y<Ȣ<���;X�D�Tx�<�y<X�S�гӻ N[��vi�@Rջ~m=�5���
<�E���<]^<��;8Գ<��<�'�����X9开�Ȼ$�ܼ VF�T��< �������W�=�\k=�輐�|�`$�HS0�(Q<��A=�<��$=��<lf���(;�HͼXZ�<�gv�45]� ��<櫈=ЃϻP���4M�<���<H<Xa}���0�0��<t�<.�=0�,� ��� ��;�@:���������
�<|ǉ<  !� G1:�C�<�ҕ�hGR� 펹���PԻ�;�K$<��=����@҈;�1��t������ϕ�(7V���c= =�)����m�HOڼ@�߼pbO<��;��;��<���<@T�:���:`�����<(�R�үK��%=*�u=^�>�	+�������:H��<�`���>��=p��-���r�^����<��<BST��< n>��=�[�<�����<��_� eϼd��<��=�ͱ<y�@�;�>Z=h�6���>D=,h�<<��<D��Z�{= �1�`U4�=�+ܼ�m=�����&$���f<8��Gc��I�<T�=����;Z> ���@��������ܼ<Р=�d��������I�=f��b��=ԡQ�2��m�=w¹� c��趁<2<�6d=�<=l�=�@Ѽ,�<�x�=�!�<T)������u�n��|��<�&~�+�h�j<�Ed<d��=p$�}2��l�l�(�\� ,`<�KнH˘<���<=$K<\"��lb�,D�<��f=@t���߽٘�= ;�2c=�,���#��@�I�a��=�
J=>2�H=X&��+*=zY=���=;�=ɍ<`v�;�?.��ǉ��m]�8��<M+�� �<=hE=p[ؼrW<<!���9Ѽ�2�:�<p��< �3�)=t���Q�<(��<�;K�<04�<50=��<p��<@�;g� = Ў��2D=p�"��z<�u����<�h<��=(�<�J<b�=������
=X��<`t5��rһ�k�;8��<�-|=�;(�;��;�<r�=ذ	����<oL8=  R8 
s;�\Y��_<�����;#=�c~��M<���<�\=n@ =$V���u<p��;�U�|j�<�c=2<�3#=@lU<Pϣ�8<�<���<̆�<�	����<��=��s�h�����)=��5=��<`;9��p�< �<�}8=R�<��;�)6<'=�2��x�{��Ԉ�@�ɺ�M�<��<|=8h< �z< r��B$o=е�;��<���<��'=�
�=`�J� +��`;������; �;�ʑ<�f�=��=TC�<P�]<�E��9^�u�>=`{�;a}=�k<�Y�< �:H��<�FS;��n=�.�:,����"���9!=��
��3�tģ��Y�����:�j!���=��=่<p�_�z�2��sW=�<<�u<�Яm<`��=��E=�(��Ӌ�hh,<�=���B�=0'ѻL���{�H
=�Y�<p>�=���=`�; 0�9PQ�<����`�=�wB��R�D�=���̟�<�U0� z;�Ӽ�������|=����r�=��> �� P�� ���0j�<��=(/!��w<
��ħ= |�0�<�v ���t���>�:�L�7��"'=`�i��hm=Xd/<t/d=`>c�݄=h��=�j�<�40<�<��0:(4ۼ b"����"�S�xF<HH<#B�=<�<�v�\����`�G��=�-���s�=ƈ���t� W�=@'
��<,
�==�@z��k�(�=0�4=�<@]������n���=}�Y=�`�:��=�=�ON��y���f<8q���=ps+<�Y�|���p�<�͒<�K�p�O<v1=p�~<X�t� �9�m<��U��j��0��;�%F���<���0���̻� ǽ;�������xL�<�Vo������1g=�w=��g�JȈ=B�'�$n�<?Q=u.=���<J�=�g�<�����8�<ֽ= �9�,=���D��<f�R=����,�<<�=P=�Q�p�뻠��<� »�λ'8=R��=dύ<`*j���U<��=�(1�Pŗ�h �<���D^�����=$[�=Z �_͂���<0�	�P}���*=�9=V�W=k�<\H��Xu¼(��<M��=F91=ɹ5=P��;���;@�_=���0l�;��$= 8�;�2;��<�=�=<2�<�N��P��;�[u=�=Fp�=
��v�	��E���Ѽö9=Xc�<��=�=��<�=>;>%b=N))=��<p�=ǳ=xn�mw���_=�}��L���i��<����E�<g��=���=�J��$Ǣ��<���=P��;���=�4}=�%�<���_=�ˉ<��=P.�;ܰ��t�B�D�<(�p��Zw������ý$������:o�W= ��<�t=���<l�Լ��u=�	�<b	M�'�=Dߦ=�"=��ý�����w�;5��=WJ��9�=�H���G��SZ��]= ���lp�=Ё����߻D����2="tk�<s=\�<x�<IX�=�M�;�>��Ԋ<��<O���Լy�ɽD��< ��7��>j��=�=ا;<���<_�=J��=H7�Z+=�=*�#=`�K��P1��(~<qu<x�(><ʽ �F���=@k;�ǉ=x��<�2=�6��D�=9dn=.L<��!=��=���;��<�B~<z��ڨ���2<4�ռ�F�=��=�v�<����I�b��=1�ҽ*��=�ȹ����o�=P%=���<z��=7��=�	��0!�;5Z=FU�= ���qO�>��=@�^��4�=�>�<�m<�/�=�_=lG[��5۽иq��8����=p�y<@��;(�5�9,�=�e =
=`6<;T�<=�'<��� ���E[;`wc; �9@z�:�{�@<�5���m)�d����Ͱ<`6� Ϩ���f<�਼h����=ܢS=�.�����=�i6�(�<���=(�	=�4�<�\�<H�6<��!�8��<�)=���|_2= ���-�<y�=�����)�<,��<�09=�.����;,N=�N���M+< #6=	'W=<��<�'��d �<�$=dTż���P�;�����׼ᙶ=�x�=Gŧ�ˎ���9<(P��`����;=4�=IiX=��;�.<�fK ��/�<��=��<1�{=`q/<�E<�/=�6����P<$==�5��@�!���@<��=`��;����Vn��a�=���<:V�=��(���ۼ@m������H=��#<ksL=Л
=@��:�K=�.!>���=�M=0��<�b�=ʌ =�F�H7ļ9:�=��v��H.�@3'������$<��>p:�=�t�;d^˼@B
<P?�= �<<��=��=�<LaмCk=�>H<�;�=X�t����z���8{�<�?� �-:�Q9��M>g�=�@�`��;L]�<x��U\�= �v<ނ1���b�"�= MP=�Ř<`��<ju= �r���E=�҆�d�/=�����==���� D��j��P��<��=�v�<����*Ź=���G�;��w=F��6�
��ޓ�`�Ͻ�3���-E=@r?�����WW�P+�<(�(<^�;���~�;�W�0���}���n�<h[<��<�gb=tt^���p��@��=@˧:���c=8�<��<��}�*1�2X=*��` J�r$(=���aS=�H=H�Z�C/=o��->�𹖽 ����=�=�G=@A���T=��l��u�&=0.<��=O��!T�8��<g��W��'��|t��S���%v�=`�$��Z��\=�8�ZH�=�j���O�;)-=J�E����=cbн����W�<���0Z���2%=�ࣽd�a="tt�Jl���A���{\V=��<��<�=�y�������+�E/V=��}=p��;ph����U�0�<P�<�x�<"=Pu�4������<��="�������P<�>�������蛽�䶼P��>;�=�_Ӽ�{��S�;v�����6=��0�P�ɼTvj�̧L���+�\���&>E=r�C�4��<8S�<�<���<�'��Pܻ"��,�����H�C!�����=��һ0��;��<?�t= ���k����~=4b����ü@��}J�p�o<`{�;�d^���ּX; =৭�,oռ��5�(���a'�|ӎ�0e»I;H=&|=Z�p��E��^_=�ix<`Ӂ�>+?��)��f=�}��*�&�E���^��ּ��<�Et<�6��ϓ����^#��Y���X�8ur�������hS��L&=��� $�<�(=�"< #E��U�{L=8V
<�=��~<D�H��T� uE;]�p؊���߻D�ռ�g۽$5�D��<N64�Ž�_�@%N�rE@����@�μȐ�<���������c< >N:a=�"���<��Ż:GH�$m,=$<z�i���= �D=�=̲n=��<�� =��<2��=�i���*=\�@��x�=�YS�@8�[�?�b=D�ͼDq�<x!o�*�t=z!7���<�z=ŏ����������h�+@���vp=0u�B~�`=��w�<p�x�0j��#�X��)e��爽�d�I����q�<ip�X�#< �';�M���#�"�I'�=�R���>�����WF=��S=��L����5��ؚ���qb�h�E<x���j=8�W=j������<����pZ������{="�s=��=����0 ���]���t���G�h�<<��#�:��=W�����6���{<�S���;��'ٶ�2笽+�����=$Q4���*�,h�<�5p;�U=zAν�C���(`=!��[�3=�Iܽ���(�`<�U�<��� V"���q�&M� g�9�"��
Ľ4ݹ�C%��H�=l�=��"���=�ۼ�r�@D���;=�5<P셼Ц������@��:��:p������Ȗ��@�: ��;�R�;��ټ���������Oļ�k�ڡ!�PKP� _̻�o=�q0�0�Z� j�;t鄽`�X<𶨼���\������R"�`z�<N
6� ��:	6����;�k�������� 
���Ӽ<و��|;��
�{��W�0� ������<@��`<V;�b9=K.��֎�� A���`�����w;P�)��ξ��u<�z��(�����䗼�*�L`|�h4#��=`�ûڙ�؉����<�K�(C��4���V��x��<`�T<�o?��̼�E��g��Yۼ�T.�@��:Yu���u��p�lg/�Ċ �x�B����Xw����@�`�<�7����<�O�� ���AA��T,� p�Г���<:��<�|� ���E��FƼ8���<J����@�Z��T�8H<D��4Rc���N�䜎<:¼ڬm���\<x�p<��:������S��=��S=���\�=j� =�+P���y�T��<> �H{"=H?m<��V=^��=|��<d��<p�-<�}_=(������<��$���:=���йh�0���N=�ꦼ�p�=�B��dB=�P��b��݆=lJ1������e~�&�����Ҳ$=��S���~�g� 5�9�^9<�Nt�7���?������/����`��T_����������k=$���f�=4�M��ŽHvr<0";��D1=UE���F什�o:�⢼��B��lĺ��ܼ.{A=�=N���$i�ۺ��ꊪ�8�c<5�S=T �<o�!=�ٱ��
�<��
�aѽ��S�l����T�·=�����ś���4<�Ҳ��M.��/׽�Zq��g�-��=�t����Q<P� 'd<���<�� ��9�=N7���4;Il����ٽ#h=h �<�TE�@y1��^� 	�:�+��:a���|�������pz�;࠭<L�����=lЪ��7N�x5��"�<Љ<,W���1<T��� �; ���ȭ	��Ȇ�,���N<@1�� ��9�V������� ��T�̼ȥ˼�)�`�� �r�ql�=�� � ֻ`��;��S��2/<����շ�PV���'���,d輖�=�����8Gd�pA�;h�<"=���G��L<h�F��cһ��I��a� K8�8糼`F��H���0O<̛Ǽ�e`<�V=�K7� E';��	;�y����p�P'�;P�v���7<�Ϋ;�.��Q��ռx�-�p��5i�p�L�;=�^.��#��c�P�|<�8b<,���@s
�!���b#=t�<������ع�z�	�> %�8��X2G��Wo�@Ե; $>�p�@�v��1 ���P
��i��B< Ln�$��<8��4�������� Q�� |�;�S�;l��<�0���������O�h�d�`�n��ǼL{�Pݻ��<H��$ �8~�
�=H�Q��MJ�x�"<PJ<� 3���c�ʒ���(��cg=��׽�d=�>�=^���*�px_� w�:$4�<�T��0/�;b>\��<�2���p׼��<�H����ӻ�|��-��@ȶ���l��&�;ܰ'=$;�<b�=4 ��hN=�R��waڽ�P�=��"�`��~���j��8�`<�3�;��S� �.�^1a�Ɵ7�4h�< ��9�d����t=�>��+��~_{��-��x�&=�o���V����/t=�:ҽ4F=�	i�p�ݽ�	�=��ս ��9B	�ҕ�HW<n����=��)< 8�;��/=ؗ<���>yb������a��,�<0�p��C��tͻ�y�<���=xk,�����D�������κk��v�(������(< 5;�A��[8����;�h�=�߻<���Y=:H&�@�W��3뼶�޽���@=>Xu �ԕ�(%^�"�Q�4�*=dʌ<@3�;𢤻8-<L�#=��̼Z�6�,���,+�u�����0�<�<H�I=�'�Hʹ�0�����c��v=���<�H�<0{g<K�<h=���<I
+=�s��H��<pn�;xg�<�읺FH=ȗ-<�������DJ�����y<`B�;0h�=��ۻH�Y���=�����=�Sc����<�
���4F�@����=�ğ���D:�< B�:e52=R.>�pe���9�< N2<�=Ժļ8���|<�Y�L˛<@I�;�8=�����/�<�b==,����� =�O=��X�9<�QD<�h�@�H;��3<��<�7����<�Ǽ@���X6<� �O��`= 
�<B�d�@険~-= .�<8;(��M���꘼�¤= l�<����X�0TK<V03�h́��T<�2ͼ�g<��<�k�<�NO�\H�<��(�=�[i:�Ķ�C�2=���<�׆<0�=�E� p�p�����<��<]�Y=x>�<���;xk=بV<`�A���<�}9<ٗ��¼��,��|z<	��4���?y�y�#=��m�`W��h8�������i��Zf�4"�,���|L=�~��m�<�G�=T:�� �����h�@< R��l��@Ǎ�!��=�I�<��A��E��������=�,n� �X<����ＤsV��ٳ<�aR<��=�[<�{t�C�=�N��.�T��=�Xk/�������-���<�sF���N����8�U���?l'=��<<#�=vz�=P�o�(S��f6D� G�r[�=�ꤽ�O<ēԼ��1=Drk� B:�${ʼ�}w�z>���y�� �x���[�D�<�`��I=�i�<�ԕ<�@`=����m��o��j ?��߰��I��i����LN�����=��=�AC=Ev����j��1ν�ۑ=��� �.�i6��P�����=���;�2��o�S=��=�q�����:,9鼫Y����L�@s�����g�=pǋ;����`�;�*�pȣ;�ƀ�x�ܼ��o;=�mq=����ES�0��;^�J��bT;��� ����= m/:�!���<���� ��9�O�=��< p-<�q�<t5�<�:)=$�F=F�Y=P7��@1�; a;|��<��=pØ=�l<<Zh+���d�U�Ж��4�<��B<M^=�a���<��%�<x
<<�
=H�%�������ϼ �:P>�X0<�ڏ=a�<@1;�G�<0+�<L�<��@c�;�~�<���<~0 =�(<�k<VG=0@F<:�<\�<hW$=��P<6�2= �Q:��t�<�}-=Nt�I"= ��;��k��Е���<��<0�<o�c=�v�0@�<��^�(o{��TH=���<~��ys�
�=��<�Ҽ�А<Hn���]/=�� ���A=�B�<���<�e*��#<
�4=�ؼ��<�q�<xQ=��T�X>�<h����=X�=�];�~=��Q=0&���p6=p6<�=``����=�<��M=�7��$��<���=��<`0�;J�o=�^= 4��ȏ.<4&�<�~�<��pU�`��;W a=�]��S�;YL�����XǼ�%���=�<P<�p^Ѽh��r<��Z�<��S<b~$=��/< �|�6@ ���Z�-�=�D�<�V=����[����=�.��l)�<b~���CI������<t���ۙ�=��B��x���<���f���Ч=`�»x5<x�\�\ܖ�XW� )����aY����3������{=j=���=���=^�P�g�H���
5=���=� V��8=d��<�B<H|3�Tnm���=�v*<��$>�/Ƚ E�`xȻ�J1����<2�5��+�<��$��P=�x=PȻK�<��<Ԓ�`�<д����_��0�A�@�+<���=�Q�=�e1�l�8�<�ɽ6��=���zU=��u����=x��<(&E��=p >��a�P�W� ��X�l<C廽�P�g�= L��=��:�;�<2$=�|<����U����C�|C)��=���=@��:گ^��,=�+���t=Թ��>����=L�
��I[�Pl�;�j߻��<�؊=�J$<0+��(:x< n;;<�=HΊ=0}�= ̨:�C�;�`�;F?=MR\=��s=@M��)�f��ВG�舃<h�N<�B�<v3=8_��D��h��<���<x�< �;� tX;������<8�%����<�m=� =@��;��;=�؛<�+�<���|M�< ��;�)�<��=(H�<�6�<N�=D�< Pu<"c=Q4=��C3=F+"��� �0�8Ի<"���4= ��;�I���Z����|��<��<��=�L?��s6= kt; �A���=�<�����1��H��<�y�;ב�$M�<���i�<�lq�4�=��i<�<��	���<��?=�H㼔�<�<H�+=��(�<�<��ӓ�=���= {�;ێ�=E�s=v=	���<(�u<IC=`�;�� =�>;q�*=,F�E64=�{�=��<�])<�Ҁ=Ʉ=0��<��<S*=�׬;N�x��aS���f<�j=��鼐H�<����-�<pO6����XM�<�)�=�< >��C��;���<��D�=J;w=�Xļ��༄^�=��<��=`( ��.=���� �=t�r���=X�U�fe�=��#��ƻ�H2�|��<s>���<��޽[k�=�J����?��=�ć� ��c�������񽚉�=��U���Ӽ 4�p����q=���=����='5�0�T�}��F��=�>�=0��<H��<t�y��\F�v����͂=XN� �"�=�<v��q���͇��
��h��<0�߻ȼԾ�=�<�{E=���<`�s;0�=3[��s��.���������= >@�d<x��<f��=��������E={~�=\y>�\��v�V��VM;�DS�0X��5Ľ2���frp�cye=Df�<��'�Z��=�Z��
]=x!���
V=��S= ��;g��=EN��i��T�� ε:�\,�d�
��@�=��!�L��=9T���Sa;��軐�<u��=��=�7X=<;�<r�P�D�m�p�<��N=���:x6<Ȼ������pr�<@�:��I<@f�Ь˼��ü�:�<0g�<µ��>��𱧻pR¼p���ں���r���˼0Ц��d�< ���N:� ����;��@��<p�������!��1$���}�x����Ce������=�Ƃ��nt<d¼��<�u<��*��
�����2�!�"]1�(=l�����0��;3<�I�<������Q	=�h�p� ��׼��2�P��;��;�����#��`�;�;��^��$�8A�|23�t/ּ�`k: t; ��;��D�v�x3�<���<����ѧ�޼�G�<�n
<�"��L��fX�������f�<X�)<����.�Zf��z�<�|��?μ��ѽ(�&� >Y�4��<�Q�@Q�:@h|;�zw<x5�~R�ȩ�<��<����H�PV����w��)���.�P�,�$�ؼ�X�� s滠�;�%��8���hk���;t	���N�����u3� [˼8T��\6�<r�=t��=�N?��n@��JǼ2�L����=�a|=�$z�<���=�>2=�k=Ő�`p�;���K>~c���
==> ����=�G� ��
0��<��=$߃��Ϲ��=��^��/n��)�=\���N���ﭽM�7�[���B�=�3�
�^����L���]�<�os=���91��
潜�ӽ�"���K�<ش ��RM��t�Xj���S��[Խ�=���_=< ��9��j�$��<�7��o�����;��S;ș�d�=���<�ѧ<X��< Z��U_�=UZ���b�P����1b=�&�=G� >g����~�Bhe=`�9�X���<�h$=�57>핇�?���@%�:����{���Օӽ찋�����g=�`�<|�9����=<�мص=�����n�<��?=�@����=��������@�仞�J����p4�.��=`g����=�,0�@4Ӻ�AP���;d�=�ql=���Wx�=�����-� �8;�<�<T�'��
��gn��.��C� �����;Fea� /�0@�������3��d��_o���.&�@��A��P鲻:p�4����׻�+t<�,�tM����;��E��*���;k�l����Ӽ�~�:�$��e�)��0i<z��@eC;�G(�x!2��0X<P�7�H~=�t����p]�._��Y1;D�Q������Ö�P��;񶻀�=��"<�ŋ�ȅS�$��XR�h~
��R�� J;�-~�Xۼ�|߼�9�8�_� ��h)!�hⷼ�y����C��� �,ᮼ�E�� =�9�s;0K��h{U<|��@�f;�ķ<��H����-�|���b#]���: "��@M�� ���V�����!o����g���2Ƽ踊��yg:(i��@j��|�� �� v��!����7��A;D�ռ�!�����,���D�¼��=��hr�l���. ���=�"�<�� �ܓ�ls�R���XB=P��6�z��
��ZW� м D���<0��<�:�=�d0�0j���
�R�j�ǭ*=	�=z��4ȑ<�F�=(i�<�w�= :���'	� ��:H�=t2��$;=�����C�;��Ž�U�\��Xis<�g�=��B�4ֽ�h=ՉཁG۽բ�=d����CS���8������g=L���Cx��fO���ȼd�=D��=Zu��񄩽2ǿ�����ڽl�����J�b�<�`�bU �4͖��?=��[��⻤���d�X==��½�I��t���xY��1����=�{߼(I�<��<p[����=����A�����p��=h�;=�v�=�6�����<��i=FWi�D<�T\��z�<���=)���~���6�iʇ��A����H�k�hb�<��p=Hӫ<���슅=6\�hMY��z���OX<�\=م���%=�ٺ��׈�X��<�̼������ 4=*Ʌ=��=:w�x(W<�ݽ(��< �6<p˅<�Z<�$p=��;p��Q:����*c���
w��`<qW����[�`���0e��D�h�𑒻��)�\�`�p��bs��Z�`S<?���ˉ<X�j�D��<ʅ������|��z� r�9|���:����<�E�xм<@�L�t����4�(l"��ļ�����Е#<�_����ͼ��;�H�< �N8�m��u�&$3�DRռƣ��ͦ�
�
>�"��=x7�<�i����۪;pD��B��p��ȑۼ�==�z�<�]Z��S"���n���=l��<�}d��O�;tE���u�;a�ѽ|z��R;��d!�x�V<p/�;JL�=�r��p����,=��
�Pp��R�`ϼw�Ž�lJ<�D���R��hs!<K1����p���e����;����\:ݼ��2�6��� l;x�'��%������b:� ����E<�B�\U��\����̑<��X�͔���]B������U`���<�q=��;����Po��pP�E��=�"L;n� ����I:�L��0 }���`<�����=�=����}��⮻B�j� "���{=�Pl� �l:YQ�=<1���J�=�Y����!��/ǻ4�<0��4_�<>��,�v��H�xx<��:@�;v�=�]��$Խ�?=Xx�� ���=,�K�H�m<�;�����V��<������j��}���bI���T=��=`T��cf���̽� ��eN��x��@��hҽ$ �����t��L0�d�<�y���&�`��<s6�����<�����,��Xto�����
y<��w=@YB���@<�6���I	�9^��]��&��`ݥ�GZ2= u���r���ڻ����=��`=s˽��G�2C����=�Ӽ�Ѹ�iB��0�m�P��E�<�@��`��o{�=΀= h�
�@	�I��*ʔ�P|��@�1�9Q�=�ܮ�X�P<{����F5���6= >�9�����nA<�ĺ<���=H�s<�E��x�C<���8��<d3�<��6�F=\%
=hZ^<�	�<�y��`v�����:��2!=�	���@�<�G;䁳<wҧ��29�@5�;
B2��'���<�L<T�� $��?瘽�Ż ����wa=@�C����;��(�[�D�=`Y;4�Լ�Q�<�,��z=Pz��7
� -ܼ���;�e�� t������/�;H�4��`T�H����%=x��<@Ȇ;�C6�hI���c��Hk��>c��ϼ�̈́�o6�=�r=�&~;�1��s=��<>�<��31<Ȧ���:�<���<�*���;��R�f��=�F�<t\�|��<�`Ǽ`I=������E�,'~��F��y=���hת=�F�� �C;X�<Ls�<(�H< ��\:ټ����4=D�߼@�; �=�"�@���P�`;&�= ~�;�@�����;�2)=l:��H޼-
�3�= 	���_N�@[=�;<_�8[�z��= �ؼB�����:\��<:a�J,==(�=X�<`L<<��(�8��=P�X<̙�p ����Z����=��=�3��R#�=4l��e��0�$<�2P�42��`�<(�ͼ������)=֭^��H�=4����8���5�,��o�'=��<�.|��ƽ��2�P�k< �E<(�-�E��=^�����>/=��8f4��p�=�YM�|��<X�
��½�9��8���Ҭ�ft�% �������f=��=��p=^������u)����K<v�I=6����^;< ��Bz� %���T�d.������A�e=� ý�\�;񣝽�I۽���/��4��<c�g=tk�����<����6�´���� �D9d�H��e�� ռ����<�	�=���=[wս�A��|ս�h�=������Ͼ���=��<ƲX�P���,�=ذ
< ����2���SA�JJ�,�̽t@��Sy�=K���d��<������"v = ���x̘�p\<|�<�Q >�-�<aֽp)<����/m=8�n���H�:��=0K˻`�;��
=�C��`�o�`�s<��ӻ��=����=z�=�0�=�}'�04޼pn<�3׼��.)g=��Y= v09"�&�EƟ��6�`Ƭ;㦀=�eλP�<�:��\�򼣙2=L�< -X��T�<0�ʼx �<�ʂ:�4�@L(�{=�I�;��;�a��8�<����6�Q��������< �$=�s�<�\Q��S�;�h=8����@�;��<���=e�X= �#�	�ͽ�=X=�DU=
_y�H=�����< �];(�@��<�ϼ���=@/z��	��w�<�B��o$=2�/��=O�Xۜ�@5;��=L}�
$�=���pX'<p��;�w=<?�<�:,����I1��Z�=�캼���<`��< �z��sM�l< /�9��=�X=27���0=uM�=�XL�`t-<�s��bq=�<�I��'=Rn1=ʺF���w���=��ƺ@�һ��=��/=*�'�r�$=n4�=Hq�<P6�옰��8��F�=�9A�@਺p�f���J� x湠YW�V��=\k��8��<�ɻ&��� p�����`.G<���<��;���$�*�=ֳ\�b�=�Z������H�\g��_�=�=p�g��X��>�c��D�<Hs�<1���=�ֽ�:�`(=G����+����=����"=���\>f�\枼Lu��Z��yM��㹂�:�m��H=���=Dy�=B1���ڽ�����r��U/=���=&�ƽ��<����Eo����<����I=0�u<���=��|� 8:�ϥ�9Wν`9K;�<���<T��<�� �{�=�+Ǽ ,�<8��)�Խ�U�<HȆ�V�6������u�@�^<h��=�$>�x�� �����ܽ���=�J콩�����ʽ�０��=��8<�����; �=�냻(��<΄��FE�J�)�j2ʽ�ц=P%b<���=�*��_�-=�gB����@�;N��ꊘ�0O˻d�&=H> 3=�߽@+�;��ٽ�=�O��V����=���(�$�t��<��ۻ *�dE�<�q�;��< R�v<>g)=�=��ǻ(�+��-�<,����<-R�=��6=@Q�>�:�k^����ܟ�<� _=8~<�v�<�7˼��k���0=|�=Ш�;8d�<�V����p�<�\5�pT<�-,=n�<��h<��?<�k�<pז�j �4{�<L��<�fJ=	L7=H�v<l�<�}�=�4���D��<�z7=�n�<�x{=�K��o��d��<�G=��g�)�R=��߼ �;�4�֕���{<m���~�=������;���<�O�� u
=�Iv��>�����k�<���<��ʼ/��="�M� �.<@v^���=L٘< �׺@}��H$���İ=�޼<�<L¿<PH�<��L���;�cR�櫩=%Q�=���Hr=��=*G��F;�H<�N�=���;��f<L��<��o=A֙����<b��=P�<��;QE=wP=�6��l!�<�x�=��<��'��w輀����!�=h����N<Taؽ�.=`�;(�0�i�)=���<ђ>�;�X��t�=6���H>�y�=Pi���ԗ��(>80<��<��;�Ҿ<������=����Hh<T5��.�E=�}=�B�= �<��3=��X>���<VM��2� =��z�Y]���S�=����f��ԁ��� ��ֽ�"H= U���:� ��9���i&�=���=������=�d�8�'�����5j>�o�=h�<��ڻ`�����ؐ��`��=d�b�8�<���=C�۽�|��6�B�o���`��<��I< �-:>j>�J0=�uy=H׹<r�3=.�=����;������6���=��=�Y<h�<�O>�ê��ل�h�<>�>/��=bK���w��W����������}4��p5߼<|�����<��<�~�����=�4x�!�M=�=�x�=�Fx=�[=/}�=�����'�;�Ժ�<<N�=��ek<��>�e�<��=�����4�<�K���I<��=�%=�	�=����
�c��})���:=�$�=@펺5�@=&k��ނ� �_�� �X� = ��;�;��œ�Jk7=��<�/K�] Ž�; ��:<�<`L�;&(��#�P��;<��<0�T<�7X;���;�5^�(U�<{=��<�
�����*I=�
1<����[+� �W=�*���S�<P*�{*h=BX_=fY"��z�;hm��г�;j��7�;=04�;��K��S�<*c=�&g���X��k�<���;$|���$��=��;@[����˻,����� ��yӼ������ǂ�(�㼰dɻ<0�<�5����<��/=�X��̯<�K�;�/u;�"/��L�`ݣ<�N�<~6�r��x�:�RO =���<�g<�7=hzӼ�]M��j��.�:T
h� ��9�h�P��<�ls�,ȫ<�[g�`y�<�fP�^2=��
�`�>����<�m�:@u��ت����ۼ_I��P��<l帼�EH�8���;ZA���2<��":<p��v����E�`Y�薱�<�⼚��� �<� va��#@=��h=��=.�#��z-� �l�B1 ��/�=ȓ�=�3d�TI�<�\>��<�V =k�;�<;�4��h;>|�%�9�$=7j�ף=�3�`��<�	����<��>0]ͻl�Q���]�T���\��A�=����\S5���e�K'�i3˽��s=�j���3��p�h�>� ���v=\�=�[���|�E���]�!x��
r�=oP=(*^��CF�T�o���L��w��<Yj=ȭK��h=|Ѭ<2L���u<�<�O:��Tt�<�� =����p�=�a|=��<�Ÿ<�hy=���=ZLd��GK�\����=X��=j>�P���%��$k�=��<�Sh��\�;�m�=m?>Xt��6�A�p�����м	ĽU7��D��-<�O==�~=�P��k#>���-J=�3v<,�=P�;=�Z�;
V�=u��N6�p��;����-2�0~<����=l�<0P�=��H���<��_���x<Q��=�A�=h��<��@=:�j��%[��-=;E=��&�x@�<��
��f��(`{� ���&B1=�o����� zk<���<H*R<Pt�y.��&$�Ѽ�@j<P�O<���Ԍ���[;�a=`r#;������<j�ء5<MdA=�S;�A��X����<��r����$~���MM=T��Wc<��?���<�Z{=L*�x� <୺�$@ż�F`���=Ⱦ��׮��r뻘��<�6�@x���%<kּp��`=ؼ���<��P��|��8�8�<���6�������5 ���Ǽ@�c�x��H'6��Q�<Kʼ[���V�;�4y�P��<���:�Ɉ�th�<��ϼ8(�<?
=�ql�B����+�(�	<4#ݼ #ٻ�/<���R�:�]��TS�F��Ȳ7��� Ez;�g��h��<�7�Љ�<(�p�<D����m��x�g��!ѻ�R����0�$�ҿ-��#@;��F�����x��ƞ� R��pWw<��M��,�����dՒ�};V=���ƻ#��7�Ud<�t�����yrq=�X'<5u�=$��hӓ�ڲ ���H�A�=[n�=��?�S�<�*/>�<�I=�6�"����E<�\�=��w�=��ʽDl�<����1'=�#��j�<-��=4�a�*���<K,��+w��Թ�=B
�P�d<�qѽ-T2���v�=t�1�#᪽tF������C	g=J��=P���Y��nt�f#��渚��8[=h�n�t����\�s�c����E���:=ܱc���<l�������==�2���U��hE�:=��>;��=Pü�8�<�)�;=��=Qѽ ������Z�=<'�=(� >����������=`�0��'y��3*���}=P��=t�n��Ы���d��JD�Z������C=�ȓ=�=<uM�I�>�F� z��`6��4=/�<"���7=�8~���e���)=δ��.���=l\�=5��=�9�=�t����<����x=�xe=u�5=�@=4�=XO6<��}�T}�<���<�g����1� !��E����!|���d�X/U<���j�P��;�ͼ��u����;��Wx����<Zk���z=P@���%�<� ���� � �9�Ω:,��<8Hμ��I�3��=L����=��W<���<�˼
M���U$� =`<+G���.�<�ͮ�0ٛ<��K=��;$��<D����S�.Z��n�,����w���.�ZT)��`�=�@�<J�ۭ�P���\5� 8Ĺ��lp����[= 3
<���zZ�6�k��jT=D�p=�~���+=��8���Ƽ/��ay<����1%�(} <H��<]x�=P����P���=��i�d��L����h;;f����h<�f��߼��k<���P�<�^½_�=�C�����Hu����Ѓ�����x����q���_�<�TV:Na�� 
;� ���1F�P�����;��C��D��<k���W��3^� �<Ԉ�= :X��><�5��j���)�>�&R<�|8��/1���<0�J�H*�3� =:���ޫ= L�~��ru?��a��L�<�+�=۬; C��x]>(h��h�"=�Dr�x퉼�==�VL=�A���=�����>ɼ�=��@c=��t���A<�d�=����^|ʽ�I�<{����P�=�82���I=3���k{�(j�@1�;��X�����Z4���ټ�Kr=��=�,�;_Ĭ����t쯽E�����k<��żx&x��9������/������:=Dk�h�{�e6�����f=���͍�����<e�<Д�<޷�=�?�� i�<��e:p��<L,�=V 	�0J�@ꇺZ��=OLQ=$m=�w�+"=t��=����eW��tZ�'S=���=S��4�Ѽ ��β���<��$�C����=�H�=��=J���W�=� n��k�P�N��)<���<����LP�<*Qf�L���~=�Gݻ�Ͻ'��=x?=6�=�\/=�쥽�r�<|���ze=\��<��� N=�~�<)�=޸w=@��:e����
��ʄ�1�"=�3���O����SC<��5�%���1<J�y��*��|=jGl���i��K�<٭���:�= >�<�i�=��ȼ������;��;��=���<=݌����=��@���=\ �<�k�;�ⲼL]�� �\��7���z��PG$<Z$�ȴ �@ڕ<n7=�W�<dּ2�Y�<	����������,��v��O!>�=j�B�����1= �����h-�Ą6��Π=�VC=}J�����q󃽓��=Ο�=fU��}��=��Dޒ�v)1����:d����e�t��<�=�:>P��;T�}�8 �=|:�<�p�; �`�`��;k4���AS=����@�Ȼ�=�������<j}����I=�ⶺ���;0ϼ\���O�<<2��
��t�:��6p=��<��н��<�����r��K�b��=dZ���� 0e���0<����'y�=�J>\�<��'=��f<"����!>��=&�'��Lܼ �:��8�;�-����S=F��|��=��5��韽�f�vj���;�c=`�м2�?��=�G���<x�c�̵�<�5=�Y캐��;o��=g!�8t���(�ھ�= ��:���r��=쏿�W���=ܚ#�w����8/=p��^=>2����b�M�\�ͼ8���D�F�:�4�ļ��=$��=��=�_��co ��tϽe�׽`�<��;B�����<��ѽe&���o�<̆�<�"�("��> �7���-=�]��Ţ�����o��}�=�w�=��ս��=` 8� <|��<N6��h�;x�X��=�=�.=0��$���R��=���=C���C>�̸*����=���<�+ս�ռ�u��h;;8�F<�gB������܉=�c'=�Ҍ=�̼��x�<�������\����;��<�����==����� �Y�S=�K<P��*�=8��<�9�=�i=��ҽH��<�`"�V��=�ɻЮF��ӡ=�&�\��=n�=hVT��+ ����+���>La=�f�� ��7p*����=Z�8���z��n�<v�y��I���b�=������d���<1���6�X=�Z@=��=��ɼ�T����; RȻ�Z=X�o=׭�����=��$��
�=�;=�2���W��x�H�<h�S�1ؙ���t<Ȝݽ ZɼxZr<Z��=__=���"�	�@����4��K��Pҽ�~%1��->�}=+U��.��ۆ=@��<q��v�:�hD�~r�=>VS=��Ͻ�؂��=���>e�=��_���=0�H��(������ԽY���T<(P�<�->@]���ф�=��w=���<P}�;�C<�.ڢ=���P!�<ۑ3=o����;l�'�D�^=��v=$�K=`gZ��ě���~=zM��X��d���=�[�<N�̽Q;=���?���+�<s�=��"��?��3�����<�٦�9�	>��>��<��=Dk�<�9X��6>T� =����w��a��[=���<E�=0�;��=���<������������hi��=Z=q��������u�=Gl��l��Hܼc}=�2�<��ȼ�<-=�W�=�`�t������2�=�;Gr�r,=��нD|�d(=�3�^��w�<`���oXf=ʿF�h1]��4�t���}��J���<�0���?�=Y:�=�=���j����̽�HѽX1�<�q =�����A/<�gj�SRٽ4R_=��C���=��x�ѼP6�;nH=1V˽3ν@����|m�f�	=j*=N$��@��<@�S��KS=��g<R�	�U�<���o4<=T6�<�5���~����=.>/#��\� ��$3�Q�l=�Ț;Q�Խ��ʼ��<�C=8F�<�k��i���bb=(H�<��=�a�������ľ�@t<��<�Ё<I�ҽ�Ɗ=��ʽ<��*�<\�������d<��<`��=NPj=��۽���P���K�=(c�� ��摢=��-�bz�=ˈ=����pF��ٹ��v���=���|_����;�K�=V���-f���=�OV�����;�=`)����?<S���^=�M=�0�=�Y�r8/���{�����v=B]�=��i�J��=������q=��3=PՉ�0�ٻ�6��8�<���k���d��<���R��8=Ԙ�=pN=X��<`.���V�;��;@��u˽��Ģ��b
>*��=w^��/�/2=fz=����R�<l��D%�=̾�<�� ���~��!�>��=���63�=�����̻�� ���ʼd1׽�;(��Y<�4�::�/>�C��E��kt=�l�=G9 =`��;���cͽBw�= "߼��<c�:=�V#���J��;��q>=���=���=�������;�K�=����Zj���v�Uc�=@$�;ZC���.= ��=P̽��k=܅�=�Iؼ���@��:��1=kP���f>[$>�~�<@��;X	<�A���9>�]V<�b�����J=�pP<H�4hS=��e<F#�=BՋ��q5�Y6M=�(�D��=D�<@�|;(fZ���=�}�:�ײ�t}�<�=0+a��Л=�'	� %<h�S<T)�<؄��=<9�<E=u,>TG�<���;̣�<l�<baS��u�=������^�NJ�Hr�ھl��= <pS�;,������]�����=�ў=hg<�x�=#�f�t��<rX��O >1�= ?� Ue��@�����pt��ҕ=��Z���7=ݩ=!ӽ���f)� ���B=�4�;<D�<3	>q�'=bE�=8m=��}=��$=�fo;x�e�L�
������r=��|=�
�<+�<V>> r�; 8���>�9�<��<�&66��'���=�<�Jɽ#C��;�`�� �9���:�G�<�e�=Jz�З�<���=Kd�=Xp�<��1=��e=A�����:=4��<L3	=����rm=F��=�G�<���=砧� )�;��5� +�;d�=8�>=j��=�s���R��,QY�<��=b֭=p��<>.X=Z�D�@��`W�la����W=Re= n<���<���=w�J=0O�;�ù����;@�9=��y=�V(=����_�w�+=���=lH�<P�	��CB<p%5���z=�|}=�7=�W� ��:��=8�	=�q���<�wz=0ݻ�F=8��<��=��=����hl<��i�i=H��vQ =�=�50�x=�<-'>=�D��@�<Į�<��M=�F޽������=�V�<Ƙe=(�,< ��:�w�:^�2���=��+�8����<��(�4�`=X�D<��6=~�=_���9=������;H��< ����=p�^<8 ��'׻�y�`j:=�cd=�GJ��hz=�Pt��a~��l���� =�)����u<���7H=��#�(��<� �b��=��Ƽ��=�F��Ȕ���=G̻@}<hI<�$9=o���4Z:=��_�hde��9<U�c=`;m�0V�<П<����4%�0K�`��;�мx�߼P`��.�!= �׺���<�3=}��=t��=�Ľ�@�	��Y=L�����=��<dJ�<�~�<�>8��<�Ҵ;"�K=(��<�����=����;>=ݻp��=B���)=�?|;��P=
��=��=xO�<hxf���-=X%�+��=0�˼\{˼�c��J�����	����<�'%<�/����;����ҍ=���=�^� ��<}����껢����=�l�=�K[��W��zH��<ͼ��P���y=�(��,=z-=��I���N<@-<��y�U�#=3@?= ��<�u�=hݪ=:�=�U=�%�=�/�=��;�B�h#P�F�=���=�>&�Q�$��}��=�I =�o�`�;p�=���=�詻�VK�x2�<d��<o͡�"�7�b��Q><<��<~g$=�ف=E�>@3<;+bt=�-o=���=`��<��A=�-=�>��`�<�k�<���|��� �<�d�= �-<��=N�.��A��`?X�0��D]>���=���<o@$=�Ȫ�2ͩ���=�r�=d�Ҽ&�$=�w2�,���/#��H�2`j=$Ӌ<@�<|g�<�P�=�=���� �v�&���L=�7S=��]=�<"�D�q���'=�<�=Ыw<"H��
=�逽��F=3�=��<\�.����;���=�La<�ץ�s�<���=� ��/K=j�=~ܓ=<X><����2�<`e�@��<�!���<�ȸ;,$h� ���`=�Ɇ��b=�� <D�<$u½jХ����=Xt�<�x=P�;��r��Z�*7l��#�;v� ��c�� x0< �A��D=�L�:�q�)E=��\�L= 8];(�]���3=����=�=xVۼ�,��P!�L�<�^<��>�@��<@?� ��:�dؼ\>�<�Q��^H<C�ؽ��=z���z�<L��u"�=h?H��dZ=0��;������C<`�� �;`�<.��=�I���="]�@���`�X<�=ȼ��L��<�_̻H0ü��F�܌�}�v=���.����J<H;^= !ڻ��<��6=���<}�!=�ݽ��̼���<�34��d�=��`=�<�g	��(>`�{<��� ��:��1; ����=Ĩ8�`��=�QZ�*�-=�ԽҠ.=����=�R�=�Ǻ<\><��:0�<d/켘��=h"���<j��ހ����$��ik<��]�^� �<��<T��< EI=�+�=(~y�Lx�<�_�|����y!<g��=<��<��<x�H�������H���fSI=,d0���<�[������&=l=ڼ�����;�er= �=���= �]:�Q�<�|�;]_=|�=���������ź`�=Y��=��1>A녽��K����=�d�<�l�<DmI��2$=�T�=,��xS�<8��< 톼��
��u^����0�<<=��=}�m=V�-=d�1>(��<x��< =�=�U��wI=J�(= ?~�pU|<�=�H:�䫽�{=���=D�g=�i=��0B��r~~��C�;�=ȑ�=̉�<� =P汻l3��ƣ=�m�=��
���p�)��7��t����ƽ�cK<�� �ݼ�η<)�2=؊3�|��<t�%�ψƽ���=��s���= ��������?<��F;���<^1�0o�<D�5�#��c��=�;����,=[�4=5��=@Ί���&[=.t=����xB=�1H��6�=�
>�/'�p&�<�����˘��i�,�(� �׽*?A�Vx4��o�=�^=Bl-������Ž 5��xf=�SW<-�M=h�=�,�<�Dֽ'���0
m�>Y9=	��=X�Ѽc#2=�):=�@�����N:f=��4�p/,���/;	=-> ��;@�R��J�=���<8e� S�;e<<%r��@���d<X�)���;���;U�=*]нqѐ=-?��8;�<�9�u��P��H�y=D������<�	I=�ɽ�j����s���� ��h��=8W��A��w��=��P^�<�<�;xE=5��= 0������%�;pz\���>@G��4��S=I=�=��X�@c����<��<W
�=&�۽7�(�%�|���Rn=Nh�=���;n���s:>�췻�  ���1�ßC=e�=�$�=�/�n��=��*;=u􉽯�=���L(�<���=@[<���X��<�}L���.��3h=�:{��p�<�GL��퍽�t��(�< ����Γ��P<Ĥ*=�=8=�y> �U��?������럼t��!Hu=��u���=P눼���2�!���Ѽ,��=��+�����.�j�LԠ�L��=�#���	�����~=2h`=4P�=JE���8R=��U<�9 =���=�y������;�
�=�0�=��2>�m��֕��t�=�3��0�=�-��%g=�Q�=��K�(yh=TM�<=������J�K���d��Q=����=�b�=�Dj<@�8>|B�<��
�lc��X�H� Ko�H^&<	}=���螚����=�^E=��F�j�=��=
��=��R=:�:�<�=���P"=�r=�6=Ғ=O1=�k:=�;�y=��=�~3�����lk�<=���8�7�<J��`�e;��;��x*��e�<8�@<�d��� =ε	�ཧ��=F���A>x�<Ts=�+x;����P�<�_���1A= J��b�J� u>,t��8�=2�p=G�s=�Y���Ͻ>@D=|�<�MȽ(�$=�n����]=���=\==��=*c��6�0��*��r;]�>J��z7��� p��đ�i�>�2�=�9���]��*6�$﬽��5=��'<4��<J��=�lS=��}����������=��=��!��S�=`@=@ٺ��C�l49=U��?�0�&<�L=t�3>4��<�����=�V=�P�;�d�<��t;�gѽ�⧻`������J�<�Pʽls�=>Oҽm·=,�Ƽ�+�<(����y��#<�=���p�h�"��=(]-<�A�c0�,����G����=��=�?����������=�?�����=�>X\<�K�<"�=T#v��+@>`ֽ<��=�h��<vo�=��<��-�c�b=���<x[>�Շ�d�8��Õ��l���^h=;�=�:����o�B>�(�6�)�� ���-�=��2=���=�Ki��n�=�;!��{�<������=��a�<"ܰ=�':��i���Ha=vo|�h��~�<4���R�=C�"M���c��/�<@��:L7�<���;�F=��N=	�>����ѽ�G/��z�����#�z=�5����<��(=j������  ��	�= �S:@p������]�<���=�Ç�����췽sw#=�>=�*�=F��禕=dv�<P��<"��=�b½:�<�p�hU>�j�=}�=�@¼��2=n��=&���`W< (�<1w=��>y���[��= �;G��H�B���׽$"����=��=��v���=�%��(Y�������A�p�
���/����=c՚�v#��Z�=L9�=b���n�=��O=M��=�7�=R�z�FS�=s���E�=�Z=|���t�=<�<��=��g=�M=��/�;0F�#&��Q�*=�:ǽ�X����P<F�m��]��2=d������v�=Y;½P�k��=B�/��c>���=ݒ�=��l�)J�����<�_*��+�=#�F=y���E� >�|ü\��=��=$4=H�m�8l佱�V=@��g+ٽȽ=�?ս���<	��=�=&�7=�C�>N�@;#�qG��ϽL`��%̏�s��l.W>��=y����*���/<�B��T=�B�;�#<)�>Њ=]+�Qg���m�U�>��>��D��=��.<8
�H�h�%�=H�e���|�`vʻ�9Y=��f>��<���RO�=�^=��<��=�������'�<�
,��F]<R=3���l��=(���x@�=���=��\=��x�z�t�8=x	4<�����8��=ܚ<h|��$<Ş������=#��=L�������!LK=��B� �<>��C>(~:<%�=hZ�=��J�p�f>#�E=Bu� '��<�<�rf=��=���=|r=��=h�<�a����轠�L;��<o#�=_�˽}�轜?%>�o�����LH*����=���<�\=.=̲�=�<)��=V,I�Nڱ=xк�1�<�i=$���۽�(=~� �Qx��@?�L�$�=f���U �b�Y�|�<�nj�M�-=��;<�CU=̍�<�>Xli<
=>��u�����̽2C%=�"v;`V�8B*=��o�ҽ��&=P���ŕ=d�<����O7v=v��=�Y��r�b������>;�{�<h��=V���*= �B:��x=6��=L����E=�y�P�=ƅ`=u�F=L��$H�=$>֘�\��<@�<0��<�>��z��Ҙ=L��<�2��<h잼�U罆�/��o�<�C�=(!i�-si=�O����}������;��[漢>U��=�ɽ<�����=	�)=@)����U=�D=.<�=��m=Z�~�"
b=W���P��=�[=��W��c�= >'<���=fA�=��W<���+�-�{�½p`=+�ɽ�����M��i=��e��a��$yA=�ݼ@�֭�=�[���W��]�=N85�'�>K��=�ħ=�GE<}q�� �7<f��L�=1��=�2���c>x`�P�=��='�<tI���B���:=�� ��ȽFM =�μ� ~��D�=��=��L=�Y缜aƼ +�::jq�(%ӽ0��|��.~���Y>���=�-��.�>����<ز��(_�<� �< ����>~nY=�H7��]\�jH���#>�I><Cؼ��=`�x�`�x���b���<��������8k��� =��q> {T<���n��=<֒=Y:=$��<Hǃ�}�(��<����T!�<�+="�|���I=X5u����={
�=�ѯ= �ֹȌ ���=�y��������~�=��!<%���H�<PS���v����=OV�=PD�MO��n���v=�P5�+H>� Y>8�><0{X<1�h=��0���k>6�,="�x�*
dtype0*'
_output_shapes
:�
�
siamese_3/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:{{`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
"siamese_3/scala1/moments/Squeeze_1Squeeze!siamese_3/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
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
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_3/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_3/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_3/scala1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
usiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Tsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
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
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_3/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
"siamese_3/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
c
siamese_3/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala1/cond/switch_tIdentitysiamese_3/scala1/cond/Switch:1*
_output_shapes
: *
T0

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
siamese_3/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_3/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
siamese_3/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_3/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/MergeMergesiamese_3/scala1/cond/Switch_3 siamese_3/scala1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
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
siamese_3/scala1/batchnorm/addAddsiamese_3/scala1/cond/Merge_1 siamese_3/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
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
 siamese_3/scala1/batchnorm/add_1Add siamese_3/scala1/batchnorm/mul_1siamese_3/scala1/batchnorm/sub*&
_output_shapes
:{{`*
T0
p
siamese_3/scala1/ReluRelu siamese_3/scala1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese_3/scala1/poll/MaxPoolMaxPoolsiamese_3/scala1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:==`*
T0
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
"siamese_3/scala2/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*
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
siamese_3/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/concatConcatV2siamese_3/scala2/Conv2Dsiamese_3/scala2/Conv2D_1siamese_3/scala2/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
�
siamese_3/scala2/AddAddsiamese_3/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:99�*
T0
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
*siamese_3/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala2/Add%siamese_3/scala2/moments/StopGradient*
T0*'
_output_shapes
:99�
�
3siamese_3/scala2/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_3/scala2/moments/varianceMean*siamese_3/scala2/moments/SquaredDifference3siamese_3/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala2/moments/SqueezeSqueezesiamese_3/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
ksiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Hsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese_3/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_3/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_3/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_3/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
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
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
siamese_3/scala2/cond/switch_tIdentitysiamese_3/scala2/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_3/scala2/cond/switch_fIdentitysiamese_3/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala2/cond/Switch_1Switch siamese_3/scala2/moments/Squeezesiamese_3/scala2/cond/pred_id*3
_class)
'%loc:@siamese_3/scala2/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese_3/scala2/cond/Switch_2Switch"siamese_3/scala2/moments/Squeeze_1siamese_3/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_3/scala2/moments/Squeeze_1
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
siamese_3/scala2/batchnorm/addAddsiamese_3/scala2/cond/Merge_1 siamese_3/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_3/scala2/batchnorm/RsqrtRsqrtsiamese_3/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala2/batchnorm/mulMul siamese_3/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_3/scala2/batchnorm/mul_1Mulsiamese_3/scala2/Addsiamese_3/scala2/batchnorm/mul*
T0*'
_output_shapes
:99�
�
 siamese_3/scala2/batchnorm/mul_2Mulsiamese_3/scala2/cond/Mergesiamese_3/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_3/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_3/scala2/batchnorm/add_1Add siamese_3/scala2/batchnorm/mul_1siamese_3/scala2/batchnorm/sub*'
_output_shapes
:99�*
T0
q
siamese_3/scala2/ReluRelu siamese_3/scala2/batchnorm/add_1*'
_output_shapes
:99�*
T0
�
siamese_3/scala2/poll/MaxPoolMaxPoolsiamese_3/scala2/Relu*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides

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
siamese_3/scala3/AddAddsiamese_3/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_3/scala3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_3/scala3/moments/meanMeansiamese_3/scala3/Add/siamese_3/scala3/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
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
!siamese_3/scala3/moments/varianceMean*siamese_3/scala3/moments/SquaredDifference3siamese_3/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala3/moments/SqueezeSqueezesiamese_3/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_3/scala3/moments/Squeeze_1Squeeze!siamese_3/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_3/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_3/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Csiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
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
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_3/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0
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
usiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
"siamese_3/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
c
siamese_3/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala3/cond/switch_tIdentitysiamese_3/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_3/scala3/cond/switch_fIdentitysiamese_3/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala3/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_3/scala3/cond/Switch_1Switch siamese_3/scala3/moments/Squeezesiamese_3/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_2Switch"siamese_3/scala3/moments/Squeeze_1siamese_3/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_3/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_3/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese_3/scala3/cond/MergeMergesiamese_3/scala3/cond/Switch_3 siamese_3/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
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
siamese_3/scala3/batchnorm/addAddsiamese_3/scala3/cond/Merge_1 siamese_3/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
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
 siamese_3/scala3/batchnorm/mul_2Mulsiamese_3/scala3/cond/Mergesiamese_3/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_3/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_3/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
"siamese_3/scala4/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_3/scala4/split_1Split"siamese_3/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
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
siamese_3/scala4/Conv2D_1Conv2Dsiamese_3/scala4/split:1siamese_3/scala4/split_1:1*
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
siamese_3/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/concatConcatV2siamese_3/scala4/Conv2Dsiamese_3/scala4/Conv2D_1siamese_3/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
 siamese_3/scala4/moments/SqueezeSqueezesiamese_3/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_3/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_3/scala4/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
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
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_3/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
siamese_3/scala4/cond/switch_fIdentitysiamese_3/scala4/cond/Switch*
T0
*
_output_shapes
: 
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
siamese_3/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_3/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
 siamese_3/scala4/batchnorm/mul_2Mulsiamese_3/scala4/cond/Mergesiamese_3/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_3/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_3/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
siamese_3/scala5/ConstConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_3/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala5/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_3/scala5/split_1Split"siamese_3/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_3/scala5/Conv2DConv2Dsiamese_3/scala5/splitsiamese_3/scala5/split_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

�
siamese_3/scala5/Conv2D_1Conv2Dsiamese_3/scala5/split:1siamese_3/scala5/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
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
score_1/split/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
score_1/Conv2D_1Conv2Dscore_1/split:1Const*
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
score_1/Conv2D_2Conv2Dscore_1/split:2Const*
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
U
score_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*&
_output_shapes
:*

Tidx0*
T0*
N
�
adjust_1/Conv2DConv2Dscore_1/concatadjust/weights/read*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:
�
Const_1Const*
dtype0*'
_output_shapes
:�*��
value��B���"���ޛ���=��l<J���@F�:�Ɇ=���<�en���]=$6��4��<��=9�����	�"eM=�����=���[= ѵ����eS�����<��׻��<=8Pʼ�5&��RK��IG=��<�3p��ۼ�������q<���<@�>|��=�VȽ=�F8�P�׽��׼dU~�6e$=�|O��^���/���j;~t� ?���<��X��E׽FF4=��h<L�D����:��>�8���]�o����>��<h'�<��<s�=��<�j̼ ����������ٽxmD<�9��m>X�=O�� ���ѕ��2=�"˴�w=���D=PB��t��<��ս��<��i�z���\_s��N�<�����L׽�\��*��Ǆ�܇�&���
<����6�:�Q����(�<VK[�
�=���=��= �<���l<�#���'��@�g;���=�����=�ڽ���<P<E���Ԃ��b���r���t=�˼�,=(�����<lb���p=)r=��Ԥ˼@,|�N�E���<��:����^
H�໡�L��� �<�/<�v�<�>��`}��&�=�&f�D)�<@=��yp<h<��F<H��<��?;l�ټp���KC<�5<d��\��<P%:<��J<�Ի��Q�xA��L�<D�����<H�伌�&=�)�<�V�<��ݼ(�f�`�<���<����G����<p���ȕ���`<�7�<�3�<>i�^ʇ� �����[�<t�f=�T�=�S=�P[;�0��(w+<�R;�}�<`7�`D�;�t�<��y=�i���=�M���H<�`Y; �b<δ=�Ң�����}#=tdp=X<p�����<�}W���<@�[=dώ<pю�,5Ǽ(�=�(���<pp����G<,ށ<Ц��p#�;bmV=0W2��5�<�K
��`�� ��✻��J�p��;lZ�<�P˻��ʻ�����yż젴<���;��I=�Z.=�#=l���P#�;@t
��;=�\���*��2�=�n=�.�$1��<!F����=|��<���w��=0��(d<�j��B���e?�a�_=�%�Ǖ�=`�k;�7�=(Uh<���:߼@3m�`��;���<�n�;d���P�;��)<��C=l��6=�jR=�����I=�M>��=D���"��):Nmܽ�U=��l�?�<����C��ħ��*%�U -�8:�<L*>�⌻���/=4���lne������= 	�����<j��*>$Iɼ���(胼ir�=�)�=dU��8d���ü o�9����$w����,��>}��=�.�P,��0�.��=���y(��
7�|H�<4>��`�ܼ�ݽp5�}���Ljս
��h贼�jI��걽�h�<�n2<�˲�+_���J���M�;��� =����4?U� �#;���<U�u=tw�<U=� <&3L������ ��h���E=�6>�{���>9ѽ^	=�ʼ�{{�=�8�n_�tP���=��< ����`J�d��<������s=[�= �*����H%
<�w� �3�İ��r��'�@Kջ����x�N<8�;��X�<~��1�U��=:��
M=��`�;p��;���<�M=8�<`b˼���8x<9_=|Ή�E�1=�˶<h�'=�ڳ��C���%ڻ\��<֡�Ȁ=����d=�K=�.�<P<�"У� �9؍+<Fo
��Yd�X�<��J�����<�=�=6�g�.���ḽ�V�;� �<>n=%׷=�%q=@�;٤����<P�Ի�.;=0�l�8,?���A=P��=�OA�t�q= �G�0��;(P����<D�=�
";@�-��j=�2E=��i;�uv�lp�<D���@}�;�`|=�\�< �������_=h[���h=Ƞ� �Y:���<��)� xh�_;�=� k�p��<�6�8�>�$����v�����^m=�@B=����8B�:l%�T�M��a�<P�G<�os=�j=��=�8޻`�9<�����9=�}X��|m�1G�=s!�=���ؔݼ����?= ,d��w�z>�5�=؊�<Y��=舽 �+��"W<.WI��ҍ<H�=_�=�a=����5<�zּ`�c��s;<��!=H�<����ｼ�=k�ｉ�>��=P�ռw�=�3�=O�y=� ��o��US�=?���=�{�� �R����<�w������t��\n�Zč�0�=pK��\ɞ�("0=�u�� ��;86����лx���V�=\���E�=T����ý �a=К�<���<X��<�i=�=��v<�<�<�jZ���6����=�8=������	�I�<����i�<����>��p�<b������r���mw��
u½L����E��T���M���H�=�Q=�#Q����0�m�px�;��&<h��<��̽Ig�= ۿ;�&�=�}�<Ƌ����ʺ��G=�c�<��F��:=@��;;�e=��=�X�=���=ܓּ
�=6eR���� �ZN
=L�F�h��=uh=� p���Ƽ��<=���5=��=(�0� x7��<��� ��: ,��s��ر}�@��<��<i�<@��;Pd��l�\�ϼ>�=�����S%=�ݼ@�I�`i)<���=[�=a�=�> ;��0�J�=��%=pNӻ�D�<��L<�f=�ط;J�<`k;���<|V�� !=��;��0=P ^=�o�<��_��L�@����;��� f��8	=h�6���X�1<�
=M\=`٫��"��=m��9k<H�<�Ȓ=*W=�Cd=$�<�I�<��<H���F)=�;�����:=��=t]��h��<|��<F�=�<��;P�< '�;�$9<|4�=�;Q<��;�VV;@�d< Y-������,=$y�< -(� L���H=h���;<@!�`%����; ̤���;VĿ=(� ��֨<�F:�H漼P]��R�`e��f�=?i=������</���bA��(�<�)�<��<d��<�g =�-<��R<����y==Аٻ�Q�9��=R&�=ZF�����z��妼@�c��3p�5��=�3�=���<��ٽF�}�5�=��g��}8������ɭ=��=�?*=DtU�H��<0Ό<HNἀY<p(�<�+=X�ܼ����d��<�7����>G��=����+�=��	�%P&=ԫ3��w�<f�=0q4<r��=T�q�`�z���<
�9�~�^����;L$+��[�=��.> �M��3=%)0=x��<w��=��N���+�،r��2�=�0����~=�+�ز���>�������8=�4�<\��=�%�<|"�=���i�<{��=�i�;f�;�p�x.�<H�hR�<�e��FEs�4��<Pʠ���=��Ȼ�g�P�O��������<"�	��DW=:�W=%�E=��R<�����s<�MA=%\F=PW_�����>P1�<�=�4����w�x��K�S=1�=j��Xmq=ɹb=�F=n�=ۑ�=�C�=�. =2=���Ĩ���R�"S=L�	��==��=���X�ۼx3�<�o>�f==:f==h�.�����=Ӽ�!<;�Ү;����Dg����<��C=��<@�c;�ӱ<�~!<����{�= i۹��W=�݀<8=�<�l<��=!m=*=�x�<�"»��<�>W=�/�:u<!=�=..=X]Y<���<`f�;�"<0���2=��%<��=��V=S�V= G; �张i�G�<�	S����:�v=�i��H�~���`=��_=�U�<T�� Ӧ�������<���<j��=C�f=��=�z<�:����<d�<@��=$��<�z��v =$/�=�����<�&U=�~=x8n<�4�<�E=4g�<�\<�q=8��<x��<�1=p®;l�������TB�<��=��~<���<H�v=��;�d�<#�o=X<���<@���'=���=�����;�i�:�ꎼ�{�@����r��4ѫ=���=�lZ<`�O<�Ѝ�����C/3=x��<��=	�'=^�=�h�<�=Y=��<��=2�<��
�p��;0�M=8���ϼ,�������0'輰]�;C��=��=�=V������[=����H����;%ܗ=re=�ӭ��H��*�=2ǀ=�Y� ��<@��;0�C��,����<(�G<'8K=<ă=��=P����i=^P1� �!=P�ǻL��<kt�=�B<P��<�⁼ �=<����T���疧���< 8���=�>�'ܻ�f=���<KN=-I�=���<T��I�=H]D��t�< (������>������b����=�;p�=!�=*��=�����=Z�=(�<�>o=��F=��a<xO����g<�I޽�]���<�W��} �=�C�<���:��N�t��Y��=d�=��=�U;�P��;ϲ4=��p;��=�X�=<��=  s�P8o�h��=�q�=�=4v���ѥ;������>=W��=���;N�P=�3�=0�<���&= A�;B��=���< ����&�����<�>=B� �;]4=v�= �v�l2����=\ =$�<��j�'f����4�=�$O���$������ȼ�j��x�u�V6g=Dy�(A8���=x=�Ԧ�`\	>@����=2�=�/u=\4�<�f�;p-=0�;`~�<xmv=�s��{�=H��5u�=��=T��<�)+<��Ż�2=�^��V�4��<��
���<��==)��=�;t;��0�<k�<ȨE=�����m����;	%�z���->\r�=�A��/���E�<P���q�<��<t�=���=��=f�3��j�
=҉�=�r�=/�T=Ԩ�<@ʸ;�;p=�۽�pJ=trx=�k���U���)=t/	>t�(=ؖm�eJ=��v=�p=IĊ=�>��`���� �P�;O�'=@_�<�q=���=���<7�y=��2>t�I=b�=8k��:�=���=v³�|j��a�=Ѓ��${��ͼ����x>=�E*>\�t=�?ڻh6s��M����=(_<E&2>���=��<d��<v��=�< �>r�=�g��μ�q�<E�; ܺ��{��䟽^�½�d5=}�c=8g<U= {@;�ƼC�=���<��輎�'=��|==g-=y���H�B�<��<|j�=|B��z�L=P�z�����NQ�$7�< p�84ڀ=�j���=�;�Y�d=B�����<�k�<��0=��=��=�ʟ��y=���<6)-�0:E�oX��о�;P���Nu>�R�=�z�=�cc=�˃=R�(=���=4`��\�<�=�T�<���<p�`kC<d�=�0�=e�Ž��W��ȹ=���;�+�=:�k=�[2=V*Q�x�>�x<=�x�;��=���=��o<��$=�w8=������pd�;V�Y����=tm�<	�o=�d�<<Z��%�=�G����!>��}���:�R�=p=�\=��=H��=�@ý]�;=�S�=�X�=&�=`r���Gd=(��b=t@=0լ;�&�=��=ګ�/｀<f8���f�=���<��]:������=��=���<���<�ʕ=XU�$z˼G=p��<��p<`�t��j��r�����<��R� �v�|�� w��딽������-=��8��r9����='�,=]彦�>h7@�a��=?��=��A=��< ����' =�^�����<���=R""�lË=�A-����=,,�=�< c�; �u�Z�=P�ۻD1��L� =t ��쐘<��)=�c�= ���`(��<H,>=��s��߂���;�pC��)��-�>*.�=�3���r���h�<�y��
�<JM=bt=~k�=;+C=��_�"���G=?S�=
��=+��=���<�&�;Ÿ0=�X�TF=
B�=x�ܼ��X�i=��> ��<�=��1�=S��=ǂx=��=،s�� ���5)�0���6=��l<�Ga=�,�=8;k<�#c=LJ>��Y=���=x�,�?�=�En=�!��P�缻؜=@IJ��}���/�T ����=��>>	`�=���:�ƼYܻ�X�=\
�<'y5>���=��<�pL<xr�=pDp<��>��A<HC���{"��}�<`�Z;|7�0�û��>�i=Bإ��b�<@�޻P���.�J=2�X���X�`��;���ĩ�=D���<p�<��B�7���*t� ]�<H'-<@��<DWּ�O
���;��=�n6� �Q�z2�x��=c�C�=nƃ=�e �`]��G6��ķ���WQ� ����<t��VE!�8�<���N�����0�廪ǽ��X�M��8Y���>�TՃ<"�=Xe�rYL���ѽ��=�V\<8�&��_<[��=\w�<r�K��JZ� ��:>�R����L��<֯
��I�=�ݝ=�/� .n�@���(=�nת�����I=`�J���=yh��X0�<�9X����>��XPJ<�$���W��nU�X�E<o	������1��T�v��n�z�=���pｰ-h<*	��	|=��<�"�����<!	��W� =I�ŽP�� �=<{h�=X�ѽ'�=�om���5�ط\<�����5�����V�'=(� �䛽<,2� �ж���<=�<=��7<d����`�N��ҏ<T�< i��M��� ��q��0��;�l�< ��9Ҭ� �<�d�<�{�4�üT��`2;@��~�=��(:��U��$��jF��u~<��|���ڼ�3ú �@�F���_�� {�<$�|t�<��E��̈<$k��Э�;��%;�=�;����L��P�什3���ӌ;X$|�\�<��<���;�Dƻ��<�i=�Ya���%����L����<|�= h<<�=�<x4�<���`Qb��m
�؛9������
� s;BNr=@����ѻ�v��`�< ��9�ڼ��z����@е�T��<lݩ<��Ⱥ̕)� ��;#�9�<���<��:��j�N�hP��ټ0@��$	������0��(w<@��:nA=`��� �!<b.���Լ@j<`O;lc���3;���О'������`������@��;����|䇼@Q�;��=������D����<�����0���=��I=J�!���<���g�4�>DK�<��νK�s=0�j�v4�ؔ<<M�������H,= А:<�=��<��=�s�<ܳ�<d��<0_k��w6<@��;��1=�5���~Ѽ#Q��*�=_㺽�3=�'<,�<Љ<�5�=f �=�;齖����ao��\�ȍ��(�l��<:�U�V�q�TU�< OռjS�ǾY�,�ɼ��%���C��h�J᤽xb<:u=h9���m��a�><��H�W�(֞����=��n=����4��0tѼ0O\� H���Ѽ�����=;��=������l��X�tJ)��<<X�<����8ޭ�੽��1��u��D���(5
��.%�Lp�<|�� ��L�<G�ӽ-�޽J=���d�*�j���f=&���pXu��q
�8�c<��b=��`�a;�M=�����i�k���O=B+�=(�P��H�=�����pռ6��z�G�*/�h�!����e"=8�<s� �v�$��<D�ͼ��<o�P=x3ʼ>�����;��Q�@�k����^ʼ�9r�(*S����� ��9�幺�%�;���丼��=�'9�P�[<X/9����;�s&�`�<X�z<���<P�7�,�-�@в��M?<8�ȼ� �<�>;�t	<��ܼ �y�<�� 3-;4�鼠:�<�#��+�<, �<���<L�P�O�,��� �M�"���&��l��<`K�������qg<�?�<�
=.#l�`&I��s�0}~���;v�=`�Q=�3/=`W�;6*U�ؤ<���L#�< ,Ҽh���h�<�h=f� *�<�����;0&v��yS:�yt< ܻ��漋�&=�B|<�4�:�`���tA<@5μ�3c;�w$=��%<���|���0��<�Ѽ Nl: -l�X�X� �}; �����j;�aL=< $�p_=<@�:��bɼ(����T��I�4I�<`o�<ܖǼDu�ZQ�47��$�;H�Y�4R�<ڰ=��=�9����H�TF�T�=`�������Ȍ=�g=Nsz������ �T��=���<��Ľ��>h=����2Gy�v*=���ļ=���m�B=F�=�|=�hM= �7< �<>ф� H�9pO�;G�=���<��C�@u����=+h���>�==�J<L^�<��Z=
��=��������d�<|���,&=�����*���T<�Ņ�@�I�h���½�d�B��=$ͣ� ���fʼ����8����x��L쎼K��x�=����=��;�����6=�ɫ<O�=<�����<���<p�;pf�;V�.�0|&���=��R=p�����H_�`���,��<��� %���B�<|������r>��c����Ӹ�7#�>�g�P-׼��^�:�=��$=�2n��lB�0�������G=h�����=Xm����w=��=�E��PJ�����=Pj$�"���h;Œ��y]=D��=��`=9��=~c��{I���D��8�������<0!��=~h=�� ��<��ڻ0`)�0�<��B= v�;��k�<<&ἠb;`�q;�}�0����p�;���<�0�<��P<`�t��z��dݤ��Z=hn��<���@�� <l:b�=��y<Ȣ<���;X�D�Tx�<�y<X�S�гӻ N[��vi�@Rջ~m=�5���
<�E���<]^<��;8Գ<��<�'�����X9开�Ȼ$�ܼ VF�T��< �������W�=�\k=�輐�|�`$�HS0�(Q<��A=�<��$=��<lf���(;�HͼXZ�<�gv�45]� ��<櫈=ЃϻP���4M�<���<H<Xa}���0�0��<t�<.�=0�,� ��� ��;�@:���������
�<|ǉ<  !� G1:�C�<�ҕ�hGR� 펹���PԻ�;�K$<��=����@҈;�1��t������ϕ�(7V���c= =�)����m�HOڼ@�߼pbO<��;��;��<���<@T�:���:`�����<(�R�үK��%=*�u=^�>�	+�������:H��<�`���>��=p��-���r�^����<��<BST��< n>��=�[�<�����<��_� eϼd��<��=�ͱ<y�@�;�>Z=h�6���>D=,h�<<��<D��Z�{= �1�`U4�=�+ܼ�m=�����&$���f<8��Gc��I�<T�=����;Z> ���@��������ܼ<Р=�d��������I�=f��b��=ԡQ�2��m�=w¹� c��趁<2<�6d=�<=l�=�@Ѽ,�<�x�=�!�<T)������u�n��|��<�&~�+�h�j<�Ed<d��=p$�}2��l�l�(�\� ,`<�KнH˘<���<=$K<\"��lb�,D�<��f=@t���߽٘�= ;�2c=�,���#��@�I�a��=�
J=>2�H=X&��+*=zY=���=;�=ɍ<`v�;�?.��ǉ��m]�8��<M+�� �<=hE=p[ؼrW<<!���9Ѽ�2�:�<p��< �3�)=t���Q�<(��<�;K�<04�<50=��<p��<@�;g� = Ў��2D=p�"��z<�u����<�h<��=(�<�J<b�=������
=X��<`t5��rһ�k�;8��<�-|=�;(�;��;�<r�=ذ	����<oL8=  R8 
s;�\Y��_<�����;#=�c~��M<���<�\=n@ =$V���u<p��;�U�|j�<�c=2<�3#=@lU<Pϣ�8<�<���<̆�<�	����<��=��s�h�����)=��5=��<`;9��p�< �<�}8=R�<��;�)6<'=�2��x�{��Ԉ�@�ɺ�M�<��<|=8h< �z< r��B$o=е�;��<���<��'=�
�=`�J� +��`;������; �;�ʑ<�f�=��=TC�<P�]<�E��9^�u�>=`{�;a}=�k<�Y�< �:H��<�FS;��n=�.�:,����"���9!=��
��3�tģ��Y�����:�j!���=��=่<p�_�z�2��sW=�<<�u<�Яm<`��=��E=�(��Ӌ�hh,<�=���B�=0'ѻL���{�H
=�Y�<p>�=���=`�; 0�9PQ�<����`�=�wB��R�D�=���̟�<�U0� z;�Ӽ�������|=����r�=��> �� P�� ���0j�<��=(/!��w<
��ħ= |�0�<�v ���t���>�:�L�7��"'=`�i��hm=Xd/<t/d=`>c�݄=h��=�j�<�40<�<��0:(4ۼ b"����"�S�xF<HH<#B�=<�<�v�\����`�G��=�-���s�=ƈ���t� W�=@'
��<,
�==�@z��k�(�=0�4=�<@]������n���=}�Y=�`�:��=�=�ON��y���f<8q���=ps+<�Y�|���p�<�͒<�K�p�O<v1=p�~<X�t� �9�m<��U��j��0��;�%F���<���0���̻� ǽ;�������xL�<�Vo������1g=�w=��g�JȈ=B�'�$n�<?Q=u.=���<J�=�g�<�����8�<ֽ= �9�,=���D��<f�R=����,�<<�=P=�Q�p�뻠��<� »�λ'8=R��=dύ<`*j���U<��=�(1�Pŗ�h �<���D^�����=$[�=Z �_͂���<0�	�P}���*=�9=V�W=k�<\H��Xu¼(��<M��=F91=ɹ5=P��;���;@�_=���0l�;��$= 8�;�2;��<�=�=<2�<�N��P��;�[u=�=Fp�=
��v�	��E���Ѽö9=Xc�<��=�=��<�=>;>%b=N))=��<p�=ǳ=xn�mw���_=�}��L���i��<����E�<g��=���=�J��$Ǣ��<���=P��;���=�4}=�%�<���_=�ˉ<��=P.�;ܰ��t�B�D�<(�p��Zw������ý$������:o�W= ��<�t=���<l�Լ��u=�	�<b	M�'�=Dߦ=�"=��ý�����w�;5��=WJ��9�=�H���G��SZ��]= ���lp�=Ё����߻D����2="tk�<s=\�<x�<IX�=�M�;�>��Ԋ<��<O���Լy�ɽD��< ��7��>j��=�=ا;<���<_�=J��=H7�Z+=�=*�#=`�K��P1��(~<qu<x�(><ʽ �F���=@k;�ǉ=x��<�2=�6��D�=9dn=.L<��!=��=���;��<�B~<z��ڨ���2<4�ռ�F�=��=�v�<����I�b��=1�ҽ*��=�ȹ����o�=P%=���<z��=7��=�	��0!�;5Z=FU�= ���qO�>��=@�^��4�=�>�<�m<�/�=�_=lG[��5۽иq��8����=p�y<@��;(�5�9,�=�e =
=`6<;T�<=�'<��� ���E[;`wc; �9@z�:�{�@<�5���m)�d����Ͱ<`6� Ϩ���f<�਼h����=ܢS=�.�����=�i6�(�<���=(�	=�4�<�\�<H�6<��!�8��<�)=���|_2= ���-�<y�=�����)�<,��<�09=�.����;,N=�N���M+< #6=	'W=<��<�'��d �<�$=dTż���P�;�����׼ᙶ=�x�=Gŧ�ˎ���9<(P��`����;=4�=IiX=��;�.<�fK ��/�<��=��<1�{=`q/<�E<�/=�6����P<$==�5��@�!���@<��=`��;����Vn��a�=���<:V�=��(���ۼ@m������H=��#<ksL=Л
=@��:�K=�.!>���=�M=0��<�b�=ʌ =�F�H7ļ9:�=��v��H.�@3'������$<��>p:�=�t�;d^˼@B
<P?�= �<<��=��=�<LaмCk=�>H<�;�=X�t����z���8{�<�?� �-:�Q9��M>g�=�@�`��;L]�<x��U\�= �v<ނ1���b�"�= MP=�Ř<`��<ju= �r���E=�҆�d�/=�����==���� D��j��P��<��=�v�<����*Ź=���G�;��w=F��6�
��ޓ�`�Ͻ�3���-E=@r?�����WW�P+�<(�(<^�;���~�;�W�0���}���n�<h[<��<�gb=tt^���p��@��=@˧:���c=8�<��<��}�*1�2X=*��` J�r$(=���aS=�H=H�Z�C/=o��->�𹖽 ����=�=�G=@A���T=��l��u�&=0.<��=O��!T�8��<g��W��'��|t��S���%v�=`�$��Z��\=�8�ZH�=�j���O�;)-=J�E����=cbн����W�<���0Z���2%=�ࣽd�a="tt�Jl���A���{\V=��<��<�=�y�������+�E/V=��}=p��;ph����U�0�<P�<�x�<"=Pu�4������<��="�������P<�>�������蛽�䶼P��>;�=�_Ӽ�{��S�;v�����6=��0�P�ɼTvj�̧L���+�\���&>E=r�C�4��<8S�<�<���<�'��Pܻ"��,�����H�C!�����=��һ0��;��<?�t= ���k����~=4b����ü@��}J�p�o<`{�;�d^���ּX; =৭�,oռ��5�(���a'�|ӎ�0e»I;H=&|=Z�p��E��^_=�ix<`Ӂ�>+?��)��f=�}��*�&�E���^��ּ��<�Et<�6��ϓ����^#��Y���X�8ur�������hS��L&=��� $�<�(=�"< #E��U�{L=8V
<�=��~<D�H��T� uE;]�p؊���߻D�ռ�g۽$5�D��<N64�Ž�_�@%N�rE@����@�μȐ�<���������c< >N:a=�"���<��Ż:GH�$m,=$<z�i���= �D=�=̲n=��<�� =��<2��=�i���*=\�@��x�=�YS�@8�[�?�b=D�ͼDq�<x!o�*�t=z!7���<�z=ŏ����������h�+@���vp=0u�B~�`=��w�<p�x�0j��#�X��)e��爽�d�I����q�<ip�X�#< �';�M���#�"�I'�=�R���>�����WF=��S=��L����5��ؚ���qb�h�E<x���j=8�W=j������<����pZ������{="�s=��=����0 ���]���t���G�h�<<��#�:��=W�����6���{<�S���;��'ٶ�2笽+�����=$Q4���*�,h�<�5p;�U=zAν�C���(`=!��[�3=�Iܽ���(�`<�U�<��� V"���q�&M� g�9�"��
Ľ4ݹ�C%��H�=l�=��"���=�ۼ�r�@D���;=�5<P셼Ц������@��:��:p������Ȗ��@�: ��;�R�;��ټ���������Oļ�k�ڡ!�PKP� _̻�o=�q0�0�Z� j�;t鄽`�X<𶨼���\������R"�`z�<N
6� ��:	6����;�k�������� 
���Ӽ<و��|;��
�{��W�0� ������<@��`<V;�b9=K.��֎�� A���`�����w;P�)��ξ��u<�z��(�����䗼�*�L`|�h4#��=`�ûڙ�؉����<�K�(C��4���V��x��<`�T<�o?��̼�E��g��Yۼ�T.�@��:Yu���u��p�lg/�Ċ �x�B����Xw����@�`�<�7����<�O�� ���AA��T,� p�Г���<:��<�|� ���E��FƼ8���<J����@�Z��T�8H<D��4Rc���N�䜎<:¼ڬm���\<x�p<��:������S��=��S=���\�=j� =�+P���y�T��<> �H{"=H?m<��V=^��=|��<d��<p�-<�}_=(������<��$���:=���йh�0���N=�ꦼ�p�=�B��dB=�P��b��݆=lJ1������e~�&�����Ҳ$=��S���~�g� 5�9�^9<�Nt�7���?������/����`��T_����������k=$���f�=4�M��ŽHvr<0";��D1=UE���F什�o:�⢼��B��lĺ��ܼ.{A=�=N���$i�ۺ��ꊪ�8�c<5�S=T �<o�!=�ٱ��
�<��
�aѽ��S�l����T�·=�����ś���4<�Ҳ��M.��/׽�Zq��g�-��=�t����Q<P� 'd<���<�� ��9�=N7���4;Il����ٽ#h=h �<�TE�@y1��^� 	�:�+��:a���|�������pz�;࠭<L�����=lЪ��7N�x5��"�<Љ<,W���1<T��� �; ���ȭ	��Ȇ�,���N<@1�� ��9�V������� ��T�̼ȥ˼�)�`�� �r�ql�=�� � ֻ`��;��S��2/<����շ�PV���'���,d輖�=�����8Gd�pA�;h�<"=���G��L<h�F��cһ��I��a� K8�8糼`F��H���0O<̛Ǽ�e`<�V=�K7� E';��	;�y����p�P'�;P�v���7<�Ϋ;�.��Q��ռx�-�p��5i�p�L�;=�^.��#��c�P�|<�8b<,���@s
�!���b#=t�<������ع�z�	�> %�8��X2G��Wo�@Ե; $>�p�@�v��1 ���P
��i��B< Ln�$��<8��4�������� Q�� |�;�S�;l��<�0���������O�h�d�`�n��ǼL{�Pݻ��<H��$ �8~�
�=H�Q��MJ�x�"<PJ<� 3���c�ʒ���(��cg=��׽�d=�>�=^���*�px_� w�:$4�<�T��0/�;b>\��<�2���p׼��<�H����ӻ�|��-��@ȶ���l��&�;ܰ'=$;�<b�=4 ��hN=�R��waڽ�P�=��"�`��~���j��8�`<�3�;��S� �.�^1a�Ɵ7�4h�< ��9�d����t=�>��+��~_{��-��x�&=�o���V����/t=�:ҽ4F=�	i�p�ݽ�	�=��ս ��9B	�ҕ�HW<n����=��)< 8�;��/=ؗ<���>yb������a��,�<0�p��C��tͻ�y�<���=xk,�����D�������κk��v�(������(< 5;�A��[8����;�h�=�߻<���Y=:H&�@�W��3뼶�޽���@=>Xu �ԕ�(%^�"�Q�4�*=dʌ<@3�;𢤻8-<L�#=��̼Z�6�,���,+�u�����0�<�<H�I=�'�Hʹ�0�����c��v=���<�H�<0{g<K�<h=���<I
+=�s��H��<pn�;xg�<�읺FH=ȗ-<�������DJ�����y<`B�;0h�=��ۻH�Y���=�����=�Sc����<�
���4F�@����=�ğ���D:�< B�:e52=R.>�pe���9�< N2<�=Ժļ8���|<�Y�L˛<@I�;�8=�����/�<�b==,����� =�O=��X�9<�QD<�h�@�H;��3<��<�7����<�Ǽ@���X6<� �O��`= 
�<B�d�@険~-= .�<8;(��M���꘼�¤= l�<����X�0TK<V03�h́��T<�2ͼ�g<��<�k�<�NO�\H�<��(�=�[i:�Ķ�C�2=���<�׆<0�=�E� p�p�����<��<]�Y=x>�<���;xk=بV<`�A���<�}9<ٗ��¼��,��|z<	��4���?y�y�#=��m�`W��h8�������i��Zf�4"�,���|L=�~��m�<�G�=T:�� �����h�@< R��l��@Ǎ�!��=�I�<��A��E��������=�,n� �X<����ＤsV��ٳ<�aR<��=�[<�{t�C�=�N��.�T��=�Xk/�������-���<�sF���N����8�U���?l'=��<<#�=vz�=P�o�(S��f6D� G�r[�=�ꤽ�O<ēԼ��1=Drk� B:�${ʼ�}w�z>���y�� �x���[�D�<�`��I=�i�<�ԕ<�@`=����m��o��j ?��߰��I��i����LN�����=��=�AC=Ev����j��1ν�ۑ=��� �.�i6��P�����=���;�2��o�S=��=�q�����:,9鼫Y����L�@s�����g�=pǋ;����`�;�*�pȣ;�ƀ�x�ܼ��o;=�mq=����ES�0��;^�J��bT;��� ����= m/:�!���<���� ��9�O�=��< p-<�q�<t5�<�:)=$�F=F�Y=P7��@1�; a;|��<��=pØ=�l<<Zh+���d�U�Ж��4�<��B<M^=�a���<��%�<x
<<�
=H�%�������ϼ �:P>�X0<�ڏ=a�<@1;�G�<0+�<L�<��@c�;�~�<���<~0 =�(<�k<VG=0@F<:�<\�<hW$=��P<6�2= �Q:��t�<�}-=Nt�I"= ��;��k��Е���<��<0�<o�c=�v�0@�<��^�(o{��TH=���<~��ys�
�=��<�Ҽ�А<Hn���]/=�� ���A=�B�<���<�e*��#<
�4=�ؼ��<�q�<xQ=��T�X>�<h����=X�=�];�~=��Q=0&���p6=p6<�=``����=�<��M=�7��$��<���=��<`0�;J�o=�^= 4��ȏ.<4&�<�~�<��pU�`��;W a=�]��S�;YL�����XǼ�%���=�<P<�p^Ѽh��r<��Z�<��S<b~$=��/< �|�6@ ���Z�-�=�D�<�V=����[����=�.��l)�<b~���CI������<t���ۙ�=��B��x���<���f���Ч=`�»x5<x�\�\ܖ�XW� )����aY����3������{=j=���=���=^�P�g�H���
5=���=� V��8=d��<�B<H|3�Tnm���=�v*<��$>�/Ƚ E�`xȻ�J1����<2�5��+�<��$��P=�x=PȻK�<��<Ԓ�`�<д����_��0�A�@�+<���=�Q�=�e1�l�8�<�ɽ6��=���zU=��u����=x��<(&E��=p >��a�P�W� ��X�l<C廽�P�g�= L��=��:�;�<2$=�|<����U����C�|C)��=���=@��:گ^��,=�+���t=Թ��>����=L�
��I[�Pl�;�j߻��<�؊=�J$<0+��(:x< n;;<�=HΊ=0}�= ̨:�C�;�`�;F?=MR\=��s=@M��)�f��ВG�舃<h�N<�B�<v3=8_��D��h��<���<x�< �;� tX;������<8�%����<�m=� =@��;��;=�؛<�+�<���|M�< ��;�)�<��=(H�<�6�<N�=D�< Pu<"c=Q4=��C3=F+"��� �0�8Ի<"���4= ��;�I���Z����|��<��<��=�L?��s6= kt; �A���=�<�����1��H��<�y�;ב�$M�<���i�<�lq�4�=��i<�<��	���<��?=�H㼔�<�<H�+=��(�<�<��ӓ�=���= {�;ێ�=E�s=v=	���<(�u<IC=`�;�� =�>;q�*=,F�E64=�{�=��<�])<�Ҁ=Ʉ=0��<��<S*=�׬;N�x��aS���f<�j=��鼐H�<����-�<pO6����XM�<�)�=�< >��C��;���<��D�=J;w=�Xļ��༄^�=��<��=`( ��.=���� �=t�r���=X�U�fe�=��#��ƻ�H2�|��<s>���<��޽[k�=�J����?��=�ć� ��c�������񽚉�=��U���Ӽ 4�p����q=���=����='5�0�T�}��F��=�>�=0��<H��<t�y��\F�v����͂=XN� �"�=�<v��q���͇��
��h��<0�߻ȼԾ�=�<�{E=���<`�s;0�=3[��s��.���������= >@�d<x��<f��=��������E={~�=\y>�\��v�V��VM;�DS�0X��5Ľ2���frp�cye=Df�<��'�Z��=�Z��
]=x!���
V=��S= ��;g��=EN��i��T�� ε:�\,�d�
��@�=��!�L��=9T���Sa;��軐�<u��=��=�7X=<;�<r�P�D�m�p�<��N=���:x6<Ȼ������pr�<@�:��I<@f�Ь˼��ü�:�<0g�<µ��>��𱧻pR¼p���ں���r���˼0Ц��d�< ���N:� ����;��@��<p�������!��1$���}�x����Ce������=�Ƃ��nt<d¼��<�u<��*��
�����2�!�"]1�(=l�����0��;3<�I�<������Q	=�h�p� ��׼��2�P��;��;�����#��`�;�;��^��$�8A�|23�t/ּ�`k: t; ��;��D�v�x3�<���<����ѧ�޼�G�<�n
<�"��L��fX�������f�<X�)<����.�Zf��z�<�|��?μ��ѽ(�&� >Y�4��<�Q�@Q�:@h|;�zw<x5�~R�ȩ�<��<����H�PV����w��)���.�P�,�$�ؼ�X�� s滠�;�%��8���hk���;t	���N�����u3� [˼8T��\6�<r�=t��=�N?��n@��JǼ2�L����=�a|=�$z�<���=�>2=�k=Ő�`p�;���K>~c���
==> ����=�G� ��
0��<��=$߃��Ϲ��=��^��/n��)�=\���N���ﭽM�7�[���B�=�3�
�^����L���]�<�os=���91��
潜�ӽ�"���K�<ش ��RM��t�Xj���S��[Խ�=���_=< ��9��j�$��<�7��o�����;��S;ș�d�=���<�ѧ<X��< Z��U_�=UZ���b�P����1b=�&�=G� >g����~�Bhe=`�9�X���<�h$=�57>핇�?���@%�:����{���Օӽ찋�����g=�`�<|�9����=<�мص=�����n�<��?=�@����=��������@�仞�J����p4�.��=`g����=�,0�@4Ӻ�AP���;d�=�ql=���Wx�=�����-� �8;�<�<T�'��
��gn��.��C� �����;Fea� /�0@�������3��d��_o���.&�@��A��P鲻:p�4����׻�+t<�,�tM����;��E��*���;k�l����Ӽ�~�:�$��e�)��0i<z��@eC;�G(�x!2��0X<P�7�H~=�t����p]�._��Y1;D�Q������Ö�P��;񶻀�=��"<�ŋ�ȅS�$��XR�h~
��R�� J;�-~�Xۼ�|߼�9�8�_� ��h)!�hⷼ�y����C��� �,ᮼ�E�� =�9�s;0K��h{U<|��@�f;�ķ<��H����-�|���b#]���: "��@M�� ���V�����!o����g���2Ƽ踊��yg:(i��@j��|�� �� v��!����7��A;D�ռ�!�����,���D�¼��=��hr�l���. ���=�"�<�� �ܓ�ls�R���XB=P��6�z��
��ZW� м D���<0��<�:�=�d0�0j���
�R�j�ǭ*=	�=z��4ȑ<�F�=(i�<�w�= :���'	� ��:H�=t2��$;=�����C�;��Ž�U�\��Xis<�g�=��B�4ֽ�h=ՉཁG۽բ�=d����CS���8������g=L���Cx��fO���ȼd�=D��=Zu��񄩽2ǿ�����ڽl�����J�b�<�`�bU �4͖��?=��[��⻤���d�X==��½�I��t���xY��1����=�{߼(I�<��<p[����=����A�����p��=h�;=�v�=�6�����<��i=FWi�D<�T\��z�<���=)���~���6�iʇ��A����H�k�hb�<��p=Hӫ<���슅=6\�hMY��z���OX<�\=م���%=�ٺ��׈�X��<�̼������ 4=*Ʌ=��=:w�x(W<�ݽ(��< �6<p˅<�Z<�$p=��;p��Q:����*c���
w��`<qW����[�`���0e��D�h�𑒻��)�\�`�p��bs��Z�`S<?���ˉ<X�j�D��<ʅ������|��z� r�9|���:����<�E�xм<@�L�t����4�(l"��ļ�����Е#<�_����ͼ��;�H�< �N8�m��u�&$3�DRռƣ��ͦ�
�
>�"��=x7�<�i����۪;pD��B��p��ȑۼ�==�z�<�]Z��S"���n���=l��<�}d��O�;tE���u�;a�ѽ|z��R;��d!�x�V<p/�;JL�=�r��p����,=��
�Pp��R�`ϼw�Ž�lJ<�D���R��hs!<K1����p���e����;����\:ݼ��2�6��� l;x�'��%������b:� ����E<�B�\U��\����̑<��X�͔���]B������U`���<�q=��;����Po��pP�E��=�"L;n� ����I:�L��0 }���`<�����=�=����}��⮻B�j� "���{=�Pl� �l:YQ�=<1���J�=�Y����!��/ǻ4�<0��4_�<>��,�v��H�xx<��:@�;v�=�]��$Խ�?=Xx�� ���=,�K�H�m<�;�����V��<������j��}���bI���T=��=`T��cf���̽� ��eN��x��@��hҽ$ �����t��L0�d�<�y���&�`��<s6�����<�����,��Xto�����
y<��w=@YB���@<�6���I	�9^��]��&��`ݥ�GZ2= u���r���ڻ����=��`=s˽��G�2C����=�Ӽ�Ѹ�iB��0�m�P��E�<�@��`��o{�=΀= h�
�@	�I��*ʔ�P|��@�1�9Q�=�ܮ�X�P<{����F5���6= >�9�����nA<�ĺ<���=H�s<�E��x�C<���8��<d3�<��6�F=\%
=hZ^<�	�<�y��`v�����:��2!=�	���@�<�G;䁳<wҧ��29�@5�;
B2��'���<�L<T�� $��?瘽�Ż ����wa=@�C����;��(�[�D�=`Y;4�Լ�Q�<�,��z=Pz��7
� -ܼ���;�e�� t������/�;H�4��`T�H����%=x��<@Ȇ;�C6�hI���c��Hk��>c��ϼ�̈́�o6�=�r=�&~;�1��s=��<>�<��31<Ȧ���:�<���<�*���;��R�f��=�F�<t\�|��<�`Ǽ`I=������E�,'~��F��y=���hת=�F�� �C;X�<Ls�<(�H< ��\:ټ����4=D�߼@�; �=�"�@���P�`;&�= ~�;�@�����;�2)=l:��H޼-
�3�= 	���_N�@[=�;<_�8[�z��= �ؼB�����:\��<:a�J,==(�=X�<`L<<��(�8��=P�X<̙�p ����Z����=��=�3��R#�=4l��e��0�$<�2P�42��`�<(�ͼ������)=֭^��H�=4����8���5�,��o�'=��<�.|��ƽ��2�P�k< �E<(�-�E��=^�����>/=��8f4��p�=�YM�|��<X�
��½�9��8���Ҭ�ft�% �������f=��=��p=^������u)����K<v�I=6����^;< ��Bz� %���T�d.������A�e=� ý�\�;񣝽�I۽���/��4��<c�g=tk�����<����6�´���� �D9d�H��e�� ռ����<�	�=���=[wս�A��|ս�h�=������Ͼ���=��<ƲX�P���,�=ذ
< ����2���SA�JJ�,�̽t@��Sy�=K���d��<������"v = ���x̘�p\<|�<�Q >�-�<aֽp)<����/m=8�n���H�:��=0K˻`�;��
=�C��`�o�`�s<��ӻ��=����=z�=�0�=�}'�04޼pn<�3׼��.)g=��Y= v09"�&�EƟ��6�`Ƭ;㦀=�eλP�<�:��\�򼣙2=L�< -X��T�<0�ʼx �<�ʂ:�4�@L(�{=�I�;��;�a��8�<����6�Q��������< �$=�s�<�\Q��S�;�h=8����@�;��<���=e�X= �#�	�ͽ�=X=�DU=
_y�H=�����< �];(�@��<�ϼ���=@/z��	��w�<�B��o$=2�/��=O�Xۜ�@5;��=L}�
$�=���pX'<p��;�w=<?�<�:,����I1��Z�=�캼���<`��< �z��sM�l< /�9��=�X=27���0=uM�=�XL�`t-<�s��bq=�<�I��'=Rn1=ʺF���w���=��ƺ@�һ��=��/=*�'�r�$=n4�=Hq�<P6�옰��8��F�=�9A�@਺p�f���J� x湠YW�V��=\k��8��<�ɻ&��� p�����`.G<���<��;���$�*�=ֳ\�b�=�Z������H�\g��_�=�=p�g��X��>�c��D�<Hs�<1���=�ֽ�:�`(=G����+����=����"=���\>f�\枼Lu��Z��yM��㹂�:�m��H=���=Dy�=B1���ڽ�����r��U/=���=&�ƽ��<����Eo����<����I=0�u<���=��|� 8:�ϥ�9Wν`9K;�<���<T��<�� �{�=�+Ǽ ,�<8��)�Խ�U�<HȆ�V�6������u�@�^<h��=�$>�x�� �����ܽ���=�J콩�����ʽ�０��=��8<�����; �=�냻(��<΄��FE�J�)�j2ʽ�ц=P%b<���=�*��_�-=�gB����@�;N��ꊘ�0O˻d�&=H> 3=�߽@+�;��ٽ�=�O��V����=���(�$�t��<��ۻ *�dE�<�q�;��< R�v<>g)=�=��ǻ(�+��-�<,����<-R�=��6=@Q�>�:�k^����ܟ�<� _=8~<�v�<�7˼��k���0=|�=Ш�;8d�<�V����p�<�\5�pT<�-,=n�<��h<��?<�k�<pז�j �4{�<L��<�fJ=	L7=H�v<l�<�}�=�4���D��<�z7=�n�<�x{=�K��o��d��<�G=��g�)�R=��߼ �;�4�֕���{<m���~�=������;���<�O�� u
=�Iv��>�����k�<���<��ʼ/��="�M� �.<@v^���=L٘< �׺@}��H$���İ=�޼<�<L¿<PH�<��L���;�cR�櫩=%Q�=���Hr=��=*G��F;�H<�N�=���;��f<L��<��o=A֙����<b��=P�<��;QE=wP=�6��l!�<�x�=��<��'��w輀����!�=h����N<Taؽ�.=`�;(�0�i�)=���<ђ>�;�X��t�=6���H>�y�=Pi���ԗ��(>80<��<��;�Ҿ<������=����Hh<T5��.�E=�}=�B�= �<��3=��X>���<VM��2� =��z�Y]���S�=����f��ԁ��� ��ֽ�"H= U���:� ��9���i&�=���=������=�d�8�'�����5j>�o�=h�<��ڻ`�����ؐ��`��=d�b�8�<���=C�۽�|��6�B�o���`��<��I< �-:>j>�J0=�uy=H׹<r�3=.�=����;������6���=��=�Y<h�<�O>�ê��ل�h�<>�>/��=bK���w��W����������}4��p5߼<|�����<��<�~�����=�4x�!�M=�=�x�=�Fx=�[=/}�=�����'�;�Ժ�<<N�=��ek<��>�e�<��=�����4�<�K���I<��=�%=�	�=����
�c��})���:=�$�=@펺5�@=&k��ނ� �_�� �X� = ��;�;��œ�Jk7=��<�/K�] Ž�; ��:<�<`L�;&(��#�P��;<��<0�T<�7X;���;�5^�(U�<{=��<�
�����*I=�
1<����[+� �W=�*���S�<P*�{*h=BX_=fY"��z�;hm��г�;j��7�;=04�;��K��S�<*c=�&g���X��k�<���;$|���$��=��;@[����˻,����� ��yӼ������ǂ�(�㼰dɻ<0�<�5����<��/=�X��̯<�K�;�/u;�"/��L�`ݣ<�N�<~6�r��x�:�RO =���<�g<�7=hzӼ�]M��j��.�:T
h� ��9�h�P��<�ls�,ȫ<�[g�`y�<�fP�^2=��
�`�>����<�m�:@u��ت����ۼ_I��P��<l帼�EH�8���;ZA���2<��":<p��v����E�`Y�薱�<�⼚��� �<� va��#@=��h=��=.�#��z-� �l�B1 ��/�=ȓ�=�3d�TI�<�\>��<�V =k�;�<;�4��h;>|�%�9�$=7j�ף=�3�`��<�	����<��>0]ͻl�Q���]�T���\��A�=����\S5���e�K'�i3˽��s=�j���3��p�h�>� ���v=\�=�[���|�E���]�!x��
r�=oP=(*^��CF�T�o���L��w��<Yj=ȭK��h=|Ѭ<2L���u<�<�O:��Tt�<�� =����p�=�a|=��<�Ÿ<�hy=���=ZLd��GK�\����=X��=j>�P���%��$k�=��<�Sh��\�;�m�=m?>Xt��6�A�p�����м	ĽU7��D��-<�O==�~=�P��k#>���-J=�3v<,�=P�;=�Z�;
V�=u��N6�p��;����-2�0~<����=l�<0P�=��H���<��_���x<Q��=�A�=h��<��@=:�j��%[��-=;E=��&�x@�<��
��f��(`{� ���&B1=�o����� zk<���<H*R<Pt�y.��&$�Ѽ�@j<P�O<���Ԍ���[;�a=`r#;������<j�ء5<MdA=�S;�A��X����<��r����$~���MM=T��Wc<��?���<�Z{=L*�x� <୺�$@ż�F`���=Ⱦ��׮��r뻘��<�6�@x���%<kּp��`=ؼ���<��P��|��8�8�<���6�������5 ���Ǽ@�c�x��H'6��Q�<Kʼ[���V�;�4y�P��<���:�Ɉ�th�<��ϼ8(�<?
=�ql�B����+�(�	<4#ݼ #ٻ�/<���R�:�]��TS�F��Ȳ7��� Ez;�g��h��<�7�Љ�<(�p�<D����m��x�g��!ѻ�R����0�$�ҿ-��#@;��F�����x��ƞ� R��pWw<��M��,�����dՒ�};V=���ƻ#��7�Ud<�t�����yrq=�X'<5u�=$��hӓ�ڲ ���H�A�=[n�=��?�S�<�*/>�<�I=�6�"����E<�\�=��w�=��ʽDl�<����1'=�#��j�<-��=4�a�*���<K,��+w��Թ�=B
�P�d<�qѽ-T2���v�=t�1�#᪽tF������C	g=J��=P���Y��nt�f#��渚��8[=h�n�t����\�s�c����E���:=ܱc���<l�������==�2���U��hE�:=��>;��=Pü�8�<�)�;=��=Qѽ ������Z�=<'�=(� >����������=`�0��'y��3*���}=P��=t�n��Ы���d��JD�Z������C=�ȓ=�=<uM�I�>�F� z��`6��4=/�<"���7=�8~���e���)=δ��.���=l\�=5��=�9�=�t����<����x=�xe=u�5=�@=4�=XO6<��}�T}�<���<�g����1� !��E����!|���d�X/U<���j�P��;�ͼ��u����;��Wx����<Zk���z=P@���%�<� ���� � �9�Ω:,��<8Hμ��I�3��=L����=��W<���<�˼
M���U$� =`<+G���.�<�ͮ�0ٛ<��K=��;$��<D����S�.Z��n�,����w���.�ZT)��`�=�@�<J�ۭ�P���\5� 8Ĺ��lp����[= 3
<���zZ�6�k��jT=D�p=�~���+=��8���Ƽ/��ay<����1%�(} <H��<]x�=P����P���=��i�d��L����h;;f����h<�f��߼��k<���P�<�^½_�=�C�����Hu����Ѓ�����x����q���_�<�TV:Na�� 
;� ���1F�P�����;��C��D��<k���W��3^� �<Ԉ�= :X��><�5��j���)�>�&R<�|8��/1���<0�J�H*�3� =:���ޫ= L�~��ru?��a��L�<�+�=۬; C��x]>(h��h�"=�Dr�x퉼�==�VL=�A���=�����>ɼ�=��@c=��t���A<�d�=����^|ʽ�I�<{����P�=�82���I=3���k{�(j�@1�;��X�����Z4���ټ�Kr=��=�,�;_Ĭ����t쯽E�����k<��żx&x��9������/������:=Dk�h�{�e6�����f=���͍�����<e�<Д�<޷�=�?�� i�<��e:p��<L,�=V 	�0J�@ꇺZ��=OLQ=$m=�w�+"=t��=����eW��tZ�'S=���=S��4�Ѽ ��β���<��$�C����=�H�=��=J���W�=� n��k�P�N��)<���<����LP�<*Qf�L���~=�Gݻ�Ͻ'��=x?=6�=�\/=�쥽�r�<|���ze=\��<��� N=�~�<)�=޸w=@��:e����
��ʄ�1�"=�3���O����SC<��5�%���1<J�y��*��|=jGl���i��K�<٭���:�= >�<�i�=��ȼ������;��;��=���<=݌����=��@���=\ �<�k�;�ⲼL]�� �\��7���z��PG$<Z$�ȴ �@ڕ<n7=�W�<dּ2�Y�<	����������,��v��O!>�=j�B�����1= �����h-�Ą6��Π=�VC=}J�����q󃽓��=Ο�=fU��}��=��Dޒ�v)1����:d����e�t��<�=�:>P��;T�}�8 �=|:�<�p�; �`�`��;k4���AS=����@�Ȼ�=�������<j}����I=�ⶺ���;0ϼ\���O�<<2��
��t�:��6p=��<��н��<�����r��K�b��=dZ���� 0e���0<����'y�=�J>\�<��'=��f<"����!>��=&�'��Lܼ �:��8�;�-����S=F��|��=��5��韽�f�vj���;�c=`�м2�?��=�G���<x�c�̵�<�5=�Y캐��;o��=g!�8t���(�ھ�= ��:���r��=쏿�W���=ܚ#�w����8/=p��^=>2����b�M�\�ͼ8���D�F�:�4�ļ��=$��=��=�_��co ��tϽe�׽`�<��;B�����<��ѽe&���o�<̆�<�"�("��> �7���-=�]��Ţ�����o��}�=�w�=��ս��=` 8� <|��<N6��h�;x�X��=�=�.=0��$���R��=���=C���C>�̸*����=���<�+ս�ռ�u��h;;8�F<�gB������܉=�c'=�Ҍ=�̼��x�<�������\����;��<�����==����� �Y�S=�K<P��*�=8��<�9�=�i=��ҽH��<�`"�V��=�ɻЮF��ӡ=�&�\��=n�=hVT��+ ����+���>La=�f�� ��7p*����=Z�8���z��n�<v�y��I���b�=������d���<1���6�X=�Z@=��=��ɼ�T����; RȻ�Z=X�o=׭�����=��$��
�=�;=�2���W��x�H�<h�S�1ؙ���t<Ȝݽ ZɼxZr<Z��=__=���"�	�@����4��K��Pҽ�~%1��->�}=+U��.��ۆ=@��<q��v�:�hD�~r�=>VS=��Ͻ�؂��=���>e�=��_���=0�H��(������ԽY���T<(P�<�->@]���ф�=��w=���<P}�;�C<�.ڢ=���P!�<ۑ3=o����;l�'�D�^=��v=$�K=`gZ��ě���~=zM��X��d���=�[�<N�̽Q;=���?���+�<s�=��"��?��3�����<�٦�9�	>��>��<��=Dk�<�9X��6>T� =����w��a��[=���<E�=0�;��=���<������������hi��=Z=q��������u�=Gl��l��Hܼc}=�2�<��ȼ�<-=�W�=�`�t������2�=�;Gr�r,=��нD|�d(=�3�^��w�<`���oXf=ʿF�h1]��4�t���}��J���<�0���?�=Y:�=�=���j����̽�HѽX1�<�q =�����A/<�gj�SRٽ4R_=��C���=��x�ѼP6�;nH=1V˽3ν@����|m�f�	=j*=N$��@��<@�S��KS=��g<R�	�U�<���o4<=T6�<�5���~����=.>/#��\� ��$3�Q�l=�Ț;Q�Խ��ʼ��<�C=8F�<�k��i���bb=(H�<��=�a�������ľ�@t<��<�Ё<I�ҽ�Ɗ=��ʽ<��*�<\�������d<��<`��=NPj=��۽���P���K�=(c�� ��摢=��-�bz�=ˈ=����pF��ٹ��v���=���|_����;�K�=V���-f���=�OV�����;�=`)����?<S���^=�M=�0�=�Y�r8/���{�����v=B]�=��i�J��=������q=��3=PՉ�0�ٻ�6��8�<���k���d��<���R��8=Ԙ�=pN=X��<`.���V�;��;@��u˽��Ģ��b
>*��=w^��/�/2=fz=����R�<l��D%�=̾�<�� ���~��!�>��=���63�=�����̻�� ���ʼd1׽�;(��Y<�4�::�/>�C��E��kt=�l�=G9 =`��;���cͽBw�= "߼��<c�:=�V#���J��;��q>=���=���=�������;�K�=����Zj���v�Uc�=@$�;ZC���.= ��=P̽��k=܅�=�Iؼ���@��:��1=kP���f>[$>�~�<@��;X	<�A���9>�]V<�b�����J=�pP<H�4hS=��e<F#�=BՋ��q5�Y6M=�(�D��=D�<@�|;(fZ���=�}�:�ײ�t}�<�=0+a��Л=�'	� %<h�S<T)�<؄��=<9�<E=u,>TG�<���;̣�<l�<baS��u�=������^�NJ�Hr�ھl��= <pS�;,������]�����=�ў=hg<�x�=#�f�t��<rX��O >1�= ?� Ue��@�����pt��ҕ=��Z���7=ݩ=!ӽ���f)� ���B=�4�;<D�<3	>q�'=bE�=8m=��}=��$=�fo;x�e�L�
������r=��|=�
�<+�<V>> r�; 8���>�9�<��<�&66��'���=�<�Jɽ#C��;�`�� �9���:�G�<�e�=Jz�З�<���=Kd�=Xp�<��1=��e=A�����:=4��<L3	=����rm=F��=�G�<���=砧� )�;��5� +�;d�=8�>=j��=�s���R��,QY�<��=b֭=p��<>.X=Z�D�@��`W�la����W=Re= n<���<���=w�J=0O�;�ù����;@�9=��y=�V(=����_�w�+=���=lH�<P�	��CB<p%5���z=�|}=�7=�W� ��:��=8�	=�q���<�wz=0ݻ�F=8��<��=��=����hl<��i�i=H��vQ =�=�50�x=�<-'>=�D��@�<Į�<��M=�F޽������=�V�<Ƙe=(�,< ��:�w�:^�2���=��+�8����<��(�4�`=X�D<��6=~�=_���9=������;H��< ����=p�^<8 ��'׻�y�`j:=�cd=�GJ��hz=�Pt��a~��l���� =�)����u<���7H=��#�(��<� �b��=��Ƽ��=�F��Ȕ���=G̻@}<hI<�$9=o���4Z:=��_�hde��9<U�c=`;m�0V�<П<����4%�0K�`��;�мx�߼P`��.�!= �׺���<�3=}��=t��=�Ľ�@�	��Y=L�����=��<dJ�<�~�<�>8��<�Ҵ;"�K=(��<�����=����;>=ݻp��=B���)=�?|;��P=
��=��=xO�<hxf���-=X%�+��=0�˼\{˼�c��J�����	����<�'%<�/����;����ҍ=���=�^� ��<}����껢����=�l�=�K[��W��zH��<ͼ��P���y=�(��,=z-=��I���N<@-<��y�U�#=3@?= ��<�u�=hݪ=:�=�U=�%�=�/�=��;�B�h#P�F�=���=�>&�Q�$��}��=�I =�o�`�;p�=���=�詻�VK�x2�<d��<o͡�"�7�b��Q><<��<~g$=�ف=E�>@3<;+bt=�-o=���=`��<��A=�-=�>��`�<�k�<���|��� �<�d�= �-<��=N�.��A��`?X�0��D]>���=���<o@$=�Ȫ�2ͩ���=�r�=d�Ҽ&�$=�w2�,���/#��H�2`j=$Ӌ<@�<|g�<�P�=�=���� �v�&���L=�7S=��]=�<"�D�q���'=�<�=Ыw<"H��
=�逽��F=3�=��<\�.����;���=�La<�ץ�s�<���=� ��/K=j�=~ܓ=<X><����2�<`e�@��<�!���<�ȸ;,$h� ���`=�Ɇ��b=�� <D�<$u½jХ����=Xt�<�x=P�;��r��Z�*7l��#�;v� ��c�� x0< �A��D=�L�:�q�)E=��\�L= 8];(�]���3=����=�=xVۼ�,��P!�L�<�^<��>�@��<@?� ��:�dؼ\>�<�Q��^H<C�ؽ��=z���z�<L��u"�=h?H��dZ=0��;������C<`�� �;`�<.��=�I���="]�@���`�X<�=ȼ��L��<�_̻H0ü��F�܌�}�v=���.����J<H;^= !ڻ��<��6=���<}�!=�ݽ��̼���<�34��d�=��`=�<�g	��(>`�{<��� ��:��1; ����=Ĩ8�`��=�QZ�*�-=�ԽҠ.=����=�R�=�Ǻ<\><��:0�<d/켘��=h"���<j��ހ����$��ik<��]�^� �<��<T��< EI=�+�=(~y�Lx�<�_�|����y!<g��=<��<��<x�H�������H���fSI=,d0���<�[������&=l=ڼ�����;�er= �=���= �]:�Q�<�|�;]_=|�=���������ź`�=Y��=��1>A녽��K����=�d�<�l�<DmI��2$=�T�=,��xS�<8��< 톼��
��u^����0�<<=��=}�m=V�-=d�1>(��<x��< =�=�U��wI=J�(= ?~�pU|<�=�H:�䫽�{=���=D�g=�i=��0B��r~~��C�;�=ȑ�=̉�<� =P汻l3��ƣ=�m�=��
���p�)��7��t����ƽ�cK<�� �ݼ�η<)�2=؊3�|��<t�%�ψƽ���=��s���= ��������?<��F;���<^1�0o�<D�5�#��c��=�;����,=[�4=5��=@Ί���&[=.t=����xB=�1H��6�=�
>�/'�p&�<�����˘��i�,�(� �׽*?A�Vx4��o�=�^=Bl-������Ž 5��xf=�SW<-�M=h�=�,�<�Dֽ'���0
m�>Y9=	��=X�Ѽc#2=�):=�@�����N:f=��4�p/,���/;	=-> ��;@�R��J�=���<8e� S�;e<<%r��@���d<X�)���;���;U�=*]нqѐ=-?��8;�<�9�u��P��H�y=D������<�	I=�ɽ�j����s���� ��h��=8W��A��w��=��P^�<�<�;xE=5��= 0������%�;pz\���>@G��4��S=I=�=��X�@c����<��<W
�=&�۽7�(�%�|���Rn=Nh�=���;n���s:>�췻�  ���1�ßC=e�=�$�=�/�n��=��*;=u􉽯�=���L(�<���=@[<���X��<�}L���.��3h=�:{��p�<�GL��퍽�t��(�< ����Γ��P<Ĥ*=�=8=�y> �U��?������럼t��!Hu=��u���=P눼���2�!���Ѽ,��=��+�����.�j�LԠ�L��=�#���	�����~=2h`=4P�=JE���8R=��U<�9 =���=�y������;�
�=�0�=��2>�m��֕��t�=�3��0�=�-��%g=�Q�=��K�(yh=TM�<=������J�K���d��Q=����=�b�=�Dj<@�8>|B�<��
�lc��X�H� Ko�H^&<	}=���螚����=�^E=��F�j�=��=
��=��R=:�:�<�=���P"=�r=�6=Ғ=O1=�k:=�;�y=��=�~3�����lk�<=���8�7�<J��`�e;��;��x*��e�<8�@<�d��� =ε	�ཧ��=F���A>x�<Ts=�+x;����P�<�_���1A= J��b�J� u>,t��8�=2�p=G�s=�Y���Ͻ>@D=|�<�MȽ(�$=�n����]=���=\==��=*c��6�0��*��r;]�>J��z7��� p��đ�i�>�2�=�9���]��*6�$﬽��5=��'<4��<J��=�lS=��}����������=��=��!��S�=`@=@ٺ��C�l49=U��?�0�&<�L=t�3>4��<�����=�V=�P�;�d�<��t;�gѽ�⧻`������J�<�Pʽls�=>Oҽm·=,�Ƽ�+�<(����y��#<�=���p�h�"��=(]-<�A�c0�,����G����=��=�?����������=�?�����=�>X\<�K�<"�=T#v��+@>`ֽ<��=�h��<vo�=��<��-�c�b=���<x[>�Շ�d�8��Õ��l���^h=;�=�:����o�B>�(�6�)�� ���-�=��2=���=�Ki��n�=�;!��{�<������=��a�<"ܰ=�':��i���Ha=vo|�h��~�<4���R�=C�"M���c��/�<@��:L7�<���;�F=��N=	�>����ѽ�G/��z�����#�z=�5����<��(=j������  ��	�= �S:@p������]�<���=�Ç�����췽sw#=�>=�*�=F��禕=dv�<P��<"��=�b½:�<�p�hU>�j�=}�=�@¼��2=n��=&���`W< (�<1w=��>y���[��= �;G��H�B���׽$"����=��=��v���=�%��(Y�������A�p�
���/����=c՚�v#��Z�=L9�=b���n�=��O=M��=�7�=R�z�FS�=s���E�=�Z=|���t�=<�<��=��g=�M=��/�;0F�#&��Q�*=�:ǽ�X����P<F�m��]��2=d������v�=Y;½P�k��=B�/��c>���=ݒ�=��l�)J�����<�_*��+�=#�F=y���E� >�|ü\��=��=$4=H�m�8l佱�V=@��g+ٽȽ=�?ս���<	��=�=&�7=�C�>N�@;#�qG��ϽL`��%̏�s��l.W>��=y����*���/<�B��T=�B�;�#<)�>Њ=]+�Qg���m�U�>��>��D��=��.<8
�H�h�%�=H�e���|�`vʻ�9Y=��f>��<���RO�=�^=��<��=�������'�<�
,��F]<R=3���l��=(���x@�=���=��\=��x�z�t�8=x	4<�����8��=ܚ<h|��$<Ş������=#��=L�������!LK=��B� �<>��C>(~:<%�=hZ�=��J�p�f>#�E=Bu� '��<�<�rf=��=���=|r=��=h�<�a����轠�L;��<o#�=_�˽}�轜?%>�o�����LH*����=���<�\=.=̲�=�<)��=V,I�Nڱ=xк�1�<�i=$���۽�(=~� �Qx��@?�L�$�=f���U �b�Y�|�<�nj�M�-=��;<�CU=̍�<�>Xli<
=>��u�����̽2C%=�"v;`V�8B*=��o�ҽ��&=P���ŕ=d�<����O7v=v��=�Y��r�b������>;�{�<h��=V���*= �B:��x=6��=L����E=�y�P�=ƅ`=u�F=L��$H�=$>֘�\��<@�<0��<�>��z��Ҙ=L��<�2��<h잼�U罆�/��o�<�C�=(!i�-si=�O����}������;��[漢>U��=�ɽ<�����=	�)=@)����U=�D=.<�=��m=Z�~�"
b=W���P��=�[=��W��c�= >'<���=fA�=��W<���+�-�{�½p`=+�ɽ�����M��i=��e��a��$yA=�ݼ@�֭�=�[���W��]�=N85�'�>K��=�ħ=�GE<}q�� �7<f��L�=1��=�2���c>x`�P�=��='�<tI���B���:=�� ��ȽFM =�μ� ~��D�=��=��L=�Y缜aƼ +�::jq�(%ӽ0��|��.~���Y>���=�-��.�>����<ز��(_�<� �< ����>~nY=�H7��]\�jH���#>�I><Cؼ��=`�x�`�x���b���<��������8k��� =��q> {T<���n��=<֒=Y:=$��<Hǃ�}�(��<����T!�<�+="�|���I=X5u����={
�=�ѯ= �ֹȌ ���=�y��������~�=��!<%���H�<PS���v����=OV�=PD�MO��n���v=�P5�+H>� Y>8�><0{X<1�h=��0���k>6�,="�x�
�
siamese_4/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*&
_output_shapes
:{{`*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
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
*siamese_4/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_4/scala1/Add%siamese_4/scala1/moments/StopGradient*
T0*&
_output_shapes
:{{`
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
"siamese_4/scala1/moments/Squeeze_1Squeeze!siamese_4/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
�
&siamese_4/scala1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_4/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_4/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
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
Hsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Fsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_4/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_4/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
usiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Nsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_4/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
c
siamese_4/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_4/scala1/cond/switch_tIdentitysiamese_4/scala1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala1/cond/switch_fIdentitysiamese_4/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_4/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_4/scala1/cond/Switch_1Switch siamese_4/scala1/moments/Squeezesiamese_4/scala1/cond/pred_id*3
_class)
'%loc:@siamese_4/scala1/moments/Squeeze* 
_output_shapes
:`:`*
T0
�
siamese_4/scala1/cond/Switch_2Switch"siamese_4/scala1/moments/Squeeze_1siamese_4/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese_4/scala1/moments/Squeeze_1
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
N*
_output_shapes

:`: *
T0
�
siamese_4/scala1/cond/Merge_1Mergesiamese_4/scala1/cond/Switch_4 siamese_4/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_4/scala1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
 siamese_4/scala1/batchnorm/mul_1Mulsiamese_4/scala1/Addsiamese_4/scala1/batchnorm/mul*
T0*&
_output_shapes
:{{`
�
 siamese_4/scala1/batchnorm/mul_2Mulsiamese_4/scala1/cond/Mergesiamese_4/scala1/batchnorm/mul*
_output_shapes
:`*
T0
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
siamese_4/scala1/ReluRelu siamese_4/scala1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
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
siamese_4/scala2/splitSplit siamese_4/scala2/split/split_dimsiamese_4/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:==0:==0*
	num_split
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
siamese_4/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
"siamese_4/scala2/moments/Squeeze_1Squeeze!siamese_4/scala2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Csiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_4/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
 siamese_4/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_4/scala2/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0
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
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_4/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
usiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
Nsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
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
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
siamese_4/scala2/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_4/scala2/cond/Switch_1Switch siamese_4/scala2/moments/Squeezesiamese_4/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_4/scala2/moments/Squeeze
�
siamese_4/scala2/cond/Switch_2Switch"siamese_4/scala2/moments/Squeeze_1siamese_4/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_4/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_4/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese_4/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_4/scala2/cond/pred_id*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_4/scala2/cond/MergeMergesiamese_4/scala2/cond/Switch_3 siamese_4/scala2/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_4/scala2/cond/Merge_1Mergesiamese_4/scala2/cond/Switch_4 siamese_4/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_4/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
siamese_4/scala2/batchnorm/mulMul siamese_4/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
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
siamese_4/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_4/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_4/scala2/batchnorm/add_1Add siamese_4/scala2/batchnorm/mul_1siamese_4/scala2/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese_4/scala2/ReluRelu siamese_4/scala2/batchnorm/add_1*
T0*'
_output_shapes
:99�
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
siamese_4/scala3/Conv2DConv2Dsiamese_4/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
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
!siamese_4/scala3/moments/varianceMean*siamese_4/scala3/moments/SquaredDifference3siamese_4/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_4/scala3/moments/SqueezeSqueezesiamese_4/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_4/scala3/moments/Squeeze_1Squeeze!siamese_4/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Nsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_4/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
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
usiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Nsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
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
Ksiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_4/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
siamese_4/scala3/cond/Switch_2Switch"siamese_4/scala3/moments/Squeeze_1siamese_4/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_4/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_4/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_4/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_4/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_4/scala3/cond/MergeMergesiamese_4/scala3/cond/Switch_3 siamese_4/scala3/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
 siamese_4/scala3/batchnorm/mul_2Mulsiamese_4/scala3/cond/Mergesiamese_4/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_4/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_4/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_4/scala3/batchnorm/add_1Add siamese_4/scala3/batchnorm/mul_1siamese_4/scala3/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_4/scala3/ReluRelu siamese_4/scala3/batchnorm/add_1*'
_output_shapes
:�*
T0
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
siamese_4/scala4/Conv2DConv2Dsiamese_4/scala4/splitsiamese_4/scala4/split_1*'
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
siamese_4/scala4/Conv2D_1Conv2Dsiamese_4/scala4/split:1siamese_4/scala4/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
^
siamese_4/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala4/concatConcatV2siamese_4/scala4/Conv2Dsiamese_4/scala4/Conv2D_1siamese_4/scala4/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
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
!siamese_4/scala4/moments/varianceMean*siamese_4/scala4/moments/SquaredDifference3siamese_4/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_4/scala4/moments/SqueezeSqueezesiamese_4/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_4/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_4/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_4/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
(siamese_4/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_4/scala4/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_4/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_4/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
"siamese_4/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_4/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_4/scala4/cond/switch_tIdentitysiamese_4/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_4/scala4/cond/switch_fIdentitysiamese_4/scala4/cond/Switch*
T0
*
_output_shapes
: 
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
siamese_4/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_4/scala4/cond/pred_id*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_4/scala4/cond/MergeMergesiamese_4/scala4/cond/Switch_3 siamese_4/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
siamese_4/scala4/batchnorm/addAddsiamese_4/scala4/cond/Merge_1 siamese_4/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_4/scala4/batchnorm/RsqrtRsqrtsiamese_4/scala4/batchnorm/add*
_output_shapes	
:�*
T0
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
 siamese_4/scala4/batchnorm/mul_2Mulsiamese_4/scala4/cond/Mergesiamese_4/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_4/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_4/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
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
siamese_4/scala5/splitSplit siamese_4/scala5/split/split_dimsiamese_4/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_4/scala5/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
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
T0*
data_formatNHWC*
strides

�
siamese_4/scala5/Conv2D_1Conv2Dsiamese_4/scala5/split:1siamese_4/scala5/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
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
score_2/Conv2DConv2Dscore_2/splitConst_1*
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
score_2/Conv2D_1Conv2Dscore_2/split:1Const_1*&
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
score_2/Conv2D_2Conv2Dscore_2/split:2Const_1*&
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
score_2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
score_2/concatConcatV2score_2/Conv2Dscore_2/Conv2D_1score_2/Conv2D_2score_2/concat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
�
adjust_2/Conv2DConv2Dscore_2/concatadjust/weights/read*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
i
adjust_2/AddAddadjust_2/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0"}+�