       �K"	   �/��Abrain.Event:2L��G?q	     U�/�	���/��A"��%
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
is_trainingConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
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
<siamese/scala1/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala1/conv/weights/Initializer/truncated_normal/stddev*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`*
T0
�
8siamese/scala1/conv/weights/Initializer/truncated_normalAdd<siamese/scala1/conv/weights/Initializer/truncated_normal/mul=siamese/scala1/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
siamese/scala1/conv/weights
VariableV2*.
_class$
" loc:@siamese/scala1/conv/weights*
	container *
shape:`*
dtype0*&
_output_shapes
:`*
shared_name 
�
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`*
use_locking(
�
 siamese/scala1/conv/weights/readIdentitysiamese/scala1/conv/weights*&
_output_shapes
:`*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *o:*
dtype0
�
=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala1/conv/weights/read*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala1/conv/weights
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
siamese/scala1/Conv2DConv2DPlaceholder_2 siamese/scala1/conv/weights/read*&
_output_shapes
:;;`*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
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
VariableV2*
shared_name *)
_class
loc:@siamese/scala1/bn/beta*
	container *
shape:`*
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/beta/AssignAssignsiamese/scala1/bn/beta(siamese/scala1/bn/beta/Initializer/Const*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta
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
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`
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
3siamese/scala1/bn/moving_variance/Initializer/ConstConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*  �?
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
&siamese/scala1/bn/moving_variance/readIdentity!siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese/scala1/moments/SqueezeSqueezesiamese/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
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
Isiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zerosConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *    *
dtype0
�
7siamese/scala1/siamese/scala1/bn/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape: 
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
<siamese/scala1/siamese/scala1/bn/moving_mean/local_step/readIdentity7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Fsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepLsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Asiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/x@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Bsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/AssignAssign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepMsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zeros*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Gsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
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
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivGsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese/scala1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

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
siamese/scala1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala1/cond/Switch_1Switchsiamese/scala1/moments/Squeezesiamese/scala1/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala1/moments/Squeeze* 
_output_shapes
:`:`
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
?siamese/scala2/conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *���<*
dtype0
�
Hsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala2/conv/weights/Initializer/truncated_normal/shape*'
_output_shapes
:0�*

seed*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
seed2w*
dtype0
�
<siamese/scala2/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala2/conv/weights/Initializer/truncated_normal/stddev*'
_output_shapes
:0�*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
8siamese/scala2/conv/weights/Initializer/truncated_normalAdd<siamese/scala2/conv/weights/Initializer/truncated_normal/mul=siamese/scala2/conv/weights/Initializer/truncated_normal/mean*'
_output_shapes
:0�*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
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
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*'
_output_shapes
:0�*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *o:*
dtype0
�
=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala2/conv/weights/read*.
_class$
" loc:@siamese/scala2/conv/weights*
_output_shapes
: *
T0
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
siamese/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
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
 siamese/scala2/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala2/split_1Split siamese/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
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
(siamese/scala2/bn/beta/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*)
_class
loc:@siamese/scala2/bn/beta*
valueB�*    
�
siamese/scala2/bn/beta
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala2/bn/beta
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
siamese/scala2/bn/beta/readIdentitysiamese/scala2/bn/beta*)
_class
loc:@siamese/scala2/bn/beta*
_output_shapes	
:�*
T0
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala2/bn/gamma*
	container 
�
siamese/scala2/bn/gamma/AssignAssignsiamese/scala2/bn/gamma)siamese/scala2/bn/gamma/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma
�
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma**
_class 
loc:@siamese/scala2/bn/gamma*
_output_shapes	
:�*
T0
�
/siamese/scala2/bn/moving_mean/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
siamese/scala2/bn/moving_mean
VariableV2*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�*
dtype0
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
-siamese/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/meanMeansiamese/scala2/Add-siamese/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala2/moments/SquaredDifferenceSquaredDifferencesiamese/scala2/Add#siamese/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
�
1siamese/scala2/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala2/moments/varianceMean(siamese/scala2/moments/SquaredDifference1siamese/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
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
$siamese/scala2/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
:siamese/scala2/siamese/scala2/bn/moving_mean/biased/AssignAssign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zeros*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zerosConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *    *
dtype0
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
>siamese/scala2/siamese/scala2/bn/moving_mean/local_step/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepIsiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
<siamese/scala2/siamese/scala2/bn/moving_mean/local_step/readIdentity7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Fsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepLsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x$siamese/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala2/siamese/scala2/bn/moving_variance/biased
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
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zeros*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(
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
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read siamese/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
ssiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Rsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala2/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala2/cond/switch_tIdentitysiamese/scala2/cond/Switch:1*
_output_shapes
: *
T0

e
siamese/scala2/cond/switch_fIdentitysiamese/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese/scala2/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala2/cond/Switch_1Switchsiamese/scala2/moments/Squeezesiamese/scala2/cond/pred_id*1
_class'
%#loc:@siamese/scala2/moments/Squeeze*"
_output_shapes
:�:�*
T0
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
siamese/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
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
siamese/scala2/batchnorm/RsqrtRsqrtsiamese/scala2/batchnorm/add*
_output_shapes	
:�*
T0
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
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<
�
Hsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala3/conv/weights/Initializer/truncated_normal/shape*

seed*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
seed2�*
dtype0*(
_output_shapes
:��
�
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
siamese/scala3/conv/weights
VariableV2*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��*
dtype0
�
"siamese/scala3/conv/weights/AssignAssignsiamese/scala3/conv/weights8siamese/scala3/conv/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
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
6siamese/scala3/conv/weights/Regularizer/l2_regularizerMul<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
,siamese/scala3/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala3/conv/biases*
valueB�*���=
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
!siamese/scala3/conv/biases/AssignAssignsiamese/scala3/conv/biases,siamese/scala3/conv/biases/Initializer/Const*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
)siamese/scala3/bn/gamma/Initializer/ConstConst*
_output_shapes	
:�**
_class 
loc:@siamese/scala3/bn/gamma*
valueB�*  �?*
dtype0
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
siamese/scala3/bn/gamma/AssignAssignsiamese/scala3/bn/gamma)siamese/scala3/bn/gamma/Initializer/Const*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala3/bn/gamma/readIdentitysiamese/scala3/bn/gamma*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
_output_shapes	
:�
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
VariableV2*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0
�
$siamese/scala3/bn/moving_mean/AssignAssignsiamese/scala3/bn/moving_mean/siamese/scala3/bn/moving_mean/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
 siamese/scala3/moments/Squeeze_1Squeezesiamese/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Isiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *    
�
7siamese/scala3/siamese/scala3/bn/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape: 
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
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readsiamese/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
isiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biased@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Fsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepLsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x$siamese/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
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
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/x@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivAsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zeros*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
<siamese/scala3/siamese/scala3/bn/moving_variance/biased/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *    
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
@siamese/scala3/siamese/scala3/bn/moving_variance/local_step/readIdentity;siamese/scala3/siamese/scala3/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read siamese/scala3/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub&siamese/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x&siamese/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
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
 siamese/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

g
siamese/scala3/cond/switch_tIdentitysiamese/scala3/cond/Switch:1*
_output_shapes
: *
T0

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
siamese/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
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
siamese/scala3/batchnorm/mulMulsiamese/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/mul_1Mulsiamese/scala3/Addsiamese/scala3/batchnorm/mul*
T0*'
_output_shapes
:

�
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
=siamese/scala4/conv/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *    *
dtype0
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
Hsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala4/conv/weights/Initializer/truncated_normal/shape*

seed*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
seed2�*
dtype0*(
_output_shapes
:��
�
<siamese/scala4/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala4/conv/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
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
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala4/conv/weights*
	container 
�
"siamese/scala4/conv/weights/AssignAssignsiamese/scala4/conv/weights8siamese/scala4/conv/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
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
,siamese/scala4/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala4/conv/biases*
valueB�*���=
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
!siamese/scala4/conv/biases/AssignAssignsiamese/scala4/conv/biases,siamese/scala4/conv/biases/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
�
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
V
siamese/scala4/ConstConst*
_output_shapes
: *
value	B :*
dtype0
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
 siamese/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
siamese/scala4/bn/gamma/AssignAssignsiamese/scala4/bn/gamma)siamese/scala4/bn/gamma/Initializer/Const**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
$siamese/scala4/bn/moving_mean/AssignAssignsiamese/scala4/bn/moving_mean/siamese/scala4/bn/moving_mean/Initializer/Const*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
"siamese/scala4/bn/moving_mean/readIdentitysiamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
3siamese/scala4/bn/moving_variance/Initializer/ConstConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*  �?*
dtype0
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
&siamese/scala4/bn/moving_variance/readIdentity!siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala4/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
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
(siamese/scala4/moments/SquaredDifferenceSquaredDifferencesiamese/scala4/Add#siamese/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
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
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala4/siamese/scala4/bn/moving_mean/biased
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container 
�
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape: 
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
isiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biased@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Asiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
siamese/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
&siamese/scala4/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9
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
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
<siamese/scala4/siamese/scala4/bn/moving_variance/biased/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biased*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Bsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/AssignAssign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepMsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zeros*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
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
ssiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Rsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Lsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepRsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Gsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x&siamese/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
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
siamese/scala4/cond/Switch_2Switch siamese/scala4/moments/Squeeze_1siamese/scala4/cond/pred_id*3
_class)
'%loc:@siamese/scala4/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
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
siamese/scala4/batchnorm/addAddsiamese/scala4/cond/Merge_1siamese/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
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
Hsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala5/conv/weights/Initializer/truncated_normal/shape*

seed*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
seed2�*
dtype0*(
_output_shapes
:��
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
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases*
	container *
shape:�
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala5/conv/biases
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
 siamese/scala5/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
\
siamese/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/concatConcatV2siamese/scala5/Conv2Dsiamese/scala5/Conv2D_1siamese/scala5/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese/scala5/AddAddsiamese/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
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
siamese/scala1_1/AddAddsiamese/scala1_1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:{{`*
T0
�
/siamese/scala1_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1_1/moments/meanMeansiamese/scala1_1/Add/siamese/scala1_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
%siamese/scala1_1/moments/StopGradientStopGradientsiamese/scala1_1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese/scala1_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1_1/Add%siamese/scala1_1/moments/StopGradient*&
_output_shapes
:{{`*
T0
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
&siamese/scala1_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese/scala1_1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0
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
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Nsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese/scala1_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese/scala1_1/cond/Switch_1Switch siamese/scala1_1/moments/Squeezesiamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese/scala1_1/moments/Squeeze
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
siamese/scala1_1/ReluRelu siamese/scala1_1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
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
siamese/scala2_1/Conv2DConv2Dsiamese/scala2_1/splitsiamese/scala2_1/split_1*
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
siamese/scala2_1/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala2_1/concatConcatV2siamese/scala2_1/Conv2Dsiamese/scala2_1/Conv2D_1siamese/scala2_1/concat/axis*
N*'
_output_shapes
:99�*

Tidx0*
T0
�
siamese/scala2_1/AddAddsiamese/scala2_1/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese/scala2_1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese/scala2_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Nsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese/scala2_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala2_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze
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
N*
_output_shapes
	:�: *
T0
�
siamese/scala2_1/cond/Merge_1Mergesiamese/scala2_1/cond/Switch_4 siamese/scala2_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala2_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
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
siamese/scala3_1/moments/meanMeansiamese/scala3_1/Add/siamese/scala3_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala3_1/moments/StopGradientStopGradientsiamese/scala3_1/moments/mean*'
_output_shapes
:�*
T0
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
 siamese/scala3_1/moments/SqueezeSqueezesiamese/scala3_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Nsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
siamese/scala3_1/cond/switch_fIdentitysiamese/scala3_1/cond/Switch*
T0
*
_output_shapes
: 
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
 siamese/scala3_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*'
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
/siamese/scala4_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4_1/moments/meanMeansiamese/scala4_1/Add/siamese/scala4_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
 siamese/scala4_1/moments/SqueezeSqueezesiamese/scala4_1/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Nsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
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
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese/scala4_1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
 siamese/scala4_1/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
siamese/scala4_1/cond/Switch_2Switch"siamese/scala4_1/moments/Squeeze_1siamese/scala4_1/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese/scala4_1/moments/Squeeze_1
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
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese/scala4_1/cond/Merge_1Mergesiamese/scala4_1/cond/Switch_4 siamese/scala4_1/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
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
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
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
score/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
score/transpose	Transposesiamese/scala5/Addscore/transpose/perm*
Tperm0*
T0*'
_output_shapes
:�
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
score/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
Y
score/split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
�
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split*
T0
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
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*&
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
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
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
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*
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
score/Conv2D_6Conv2Dscore/split_1:6score/split:6*
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
score/Conv2D_7Conv2Dscore/split_1:7score/split:7*
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
score/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
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
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@adjust/biases*
	container *
shape:
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
(adjust/biases/Regularizer/l2_regularizerMul.adjust/biases/Regularizer/l2_regularizer/scale/adjust/biases/Regularizer/l2_regularizer/L2Loss* 
_class
loc:@adjust/biases*
_output_shapes
: *
T0
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
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:,*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save/Assign_2Assignsiamese/scala1/bn/betasave/RestoreV2:2*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2:5*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
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
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2:7*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`*
use_locking(*
T0
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
save/Assign_14Assignsiamese/scala2/bn/moving_meansave/RestoreV2:14*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2:16*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave/RestoreV2:20*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2:24*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_25Assign!siamese/scala3/bn/moving_variancesave/RestoreV2:25*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2:26*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave/RestoreV2:28*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2:30*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2:31*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2:32*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�
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
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave/RestoreV2:39*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
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
save/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave/RestoreV2:41*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
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
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:;;`
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
siamese_1/scala1/moments/meanMeansiamese_1/scala1/Add/siamese_1/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
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
!siamese_1/scala1/moments/varianceMean*siamese_1/scala1/moments/SquaredDifference3siamese_1/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese_1/scala1/moments/SqueezeSqueezesiamese_1/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_1/scala1/moments/Squeeze_1Squeeze!siamese_1/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_1/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Nsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
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
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_1/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
Nsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_1/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
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
siamese_1/scala1/cond/switch_fIdentitysiamese_1/scala1/cond/Switch*
_output_shapes
: *
T0

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
siamese_1/scala1/cond/Switch_2Switch"siamese_1/scala1/moments/Squeeze_1siamese_1/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*5
_class+
)'loc:@siamese_1/scala1/moments/Squeeze_1
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
siamese_1/scala1/cond/MergeMergesiamese_1/scala1/cond/Switch_3 siamese_1/scala1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
�
siamese_1/scala1/cond/Merge_1Mergesiamese_1/scala1/cond/Switch_4 siamese_1/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
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
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*
paddingVALID*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides
*
ksize

X
siamese_1/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_1/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/splitSplit siamese_1/scala2/split/split_dimsiamese_1/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
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
siamese_1/scala2/Conv2DConv2Dsiamese_1/scala2/splitsiamese_1/scala2/split_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

^
siamese_1/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_1/scala2/AddAddsiamese_1/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
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
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/mean*'
_output_shapes
:�*
T0
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
!siamese_1/scala2/moments/varianceMean*siamese_1/scala2/moments/SquaredDifference3siamese_1/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
ksiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Isiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
c
siamese_1/scala2/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala2/cond/switch_tIdentitysiamese_1/scala2/cond/Switch:1*
_output_shapes
: *
T0

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
siamese_1/scala2/batchnorm/addAddsiamese_1/scala2/cond/Merge_1 siamese_1/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala2/batchnorm/RsqrtRsqrtsiamese_1/scala2/batchnorm/add*
T0*
_output_shapes	
:�
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
siamese_1/scala2/ReluRelu siamese_1/scala2/batchnorm/add_1*
T0*'
_output_shapes
:�
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
siamese_1/scala3/AddAddsiamese_1/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
�
/siamese_1/scala3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_1/scala3/moments/meanMeansiamese_1/scala3/Add/siamese_1/scala3/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala3/moments/StopGradientStopGradientsiamese_1/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala3/Add%siamese_1/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
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
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_1/scala3/moments/Squeeze_1Squeeze!siamese_1/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Csiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Isiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
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
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
siamese_1/scala3/cond/switch_fIdentitysiamese_1/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala3/cond/Switch_1Switch siamese_1/scala3/moments/Squeezesiamese_1/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_1/scala3/moments/Squeeze
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
siamese_1/scala3/batchnorm/mulMul siamese_1/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_1/scala3/ReluRelu siamese_1/scala3/batchnorm/add_1*'
_output_shapes
:

�*
T0
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
siamese_1/scala4/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
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
*siamese_1/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala4/Add%siamese_1/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
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
&siamese_1/scala4/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_1/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_1/scala4/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
Hsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
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
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
 siamese_1/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_1/scala4/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_1/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
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
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
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
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
siamese_1/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
siamese_1/scala4/cond/Merge_1Mergesiamese_1/scala4/cond/Switch_4 siamese_1/scala4/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
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
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala4/batchnorm/add_1Add siamese_1/scala4/batchnorm/mul_1siamese_1/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_1/scala4/ReluRelu siamese_1/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
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
siamese_1/scala5/split_1Split"siamese_1/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*
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
shape:��*
dtype0*(
_output_shapes
:��
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
is_training_2Const*
_output_shapes
: *
value	B
 Z *
dtype0

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
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
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
save_1/Assign_1Assignadjust/weightssave_1/RestoreV2:1*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
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
save_1/Assign_3Assignsiamese/scala1/bn/gammasave_1/RestoreV2:3*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_4Assignsiamese/scala1/bn/moving_meansave_1/RestoreV2:4*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
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
save_1/Assign_6Assignsiamese/scala1/conv/biasessave_1/RestoreV2:6*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`
�
save_1/Assign_7Assignsiamese/scala1/conv/weightssave_1/RestoreV2:7*
validate_shape(*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
save_1/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave_1/RestoreV2:8*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save_1/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave_1/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
save_1/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave_1/RestoreV2:10*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
save_1/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave_1/RestoreV2:11*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_12Assignsiamese/scala2/bn/betasave_1/RestoreV2:12*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta
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
save_1/Assign_14Assignsiamese/scala2/bn/moving_meansave_1/RestoreV2:14*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_15Assign!siamese/scala2/bn/moving_variancesave_1/RestoreV2:15*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave_1/RestoreV2:18*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave_1/RestoreV2:19*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
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
save_1/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave_1/RestoreV2:21*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_22Assignsiamese/scala3/bn/betasave_1/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_23Assignsiamese/scala3/bn/gammasave_1/RestoreV2:23*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma
�
save_1/Assign_24Assignsiamese/scala3/bn/moving_meansave_1/RestoreV2:24*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save_1/Assign_25Assign!siamese/scala3/bn/moving_variancesave_1/RestoreV2:25*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_26Assignsiamese/scala3/conv/biasessave_1/RestoreV2:26*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save_1/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave_1/RestoreV2:28*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave_1/RestoreV2:29*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
save_1/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave_1/RestoreV2:30*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_33Assignsiamese/scala4/bn/gammasave_1/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_34Assignsiamese/scala4/bn/moving_meansave_1/RestoreV2:34*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
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
save_1/Assign_36Assignsiamese/scala4/conv/biasessave_1/RestoreV2:36*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
�
save_1/Assign_37Assignsiamese/scala4/conv/weightssave_1/RestoreV2:37*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
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
save_1/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave_1/RestoreV2:39*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
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
save_1/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave_1/RestoreV2:41*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
siamese_2/scala1/Conv2DConv2DPlaceholder_4 siamese/scala1/conv/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:;;`*
	dilations

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
*siamese_2/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala1/Add%siamese_2/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
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
&siamese_2/scala1/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0
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
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
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
Hsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_2/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
siamese_2/scala1/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

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
siamese_2/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
siamese_2/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_2/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
�
siamese_2/scala1/cond/MergeMergesiamese_2/scala1/cond/Switch_3 siamese_2/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_2/scala1/cond/Merge_1Mergesiamese_2/scala1/cond/Switch_4 siamese_2/scala1/cond/Switch_2:1*
N*
_output_shapes

:`: *
T0
e
 siamese_2/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
 siamese_2/scala1/batchnorm/mul_1Mulsiamese_2/scala1/Addsiamese_2/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
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
siamese_2/scala1/ReluRelu siamese_2/scala1/batchnorm/add_1*
T0*&
_output_shapes
:;;`
�
siamese_2/scala1/poll/MaxPoolMaxPoolsiamese_2/scala1/Relu*
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
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*'
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
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

^
siamese_2/scala2/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
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
/siamese_2/scala2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_2/scala2/moments/meanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
!siamese_2/scala2/moments/varianceMean*siamese_2/scala2/moments/SquaredDifference3siamese_2/scala2/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_2/scala2/moments/SqueezeSqueezesiamese_2/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_2/scala2/moments/Squeeze_1Squeeze!siamese_2/scala2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_2/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    
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
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_2/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
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
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
siamese_2/scala2/cond/switch_tIdentitysiamese_2/scala2/cond/Switch:1*
_output_shapes
: *
T0

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
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�*
T0
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
 siamese_2/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
siamese_2/scala3/Conv2DConv2Dsiamese_2/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�*
	dilations
*
T0
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
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
�
3siamese_2/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_2/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
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
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
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
"siamese_2/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese_2/scala3/cond/MergeMergesiamese_2/scala3/cond/Switch_3 siamese_2/scala3/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
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
 siamese_2/scala4/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese_2/scala4/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_2/scala4/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
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
/siamese_2/scala4/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_2/scala4/moments/meanMeansiamese_2/scala4/Add/siamese_2/scala4/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_2/scala4/moments/StopGradientStopGradientsiamese_2/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_2/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
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
 siamese_2/scala4/moments/SqueezeSqueezesiamese_2/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_2/scala4/moments/Squeeze_1Squeeze!siamese_2/scala4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_2/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_2/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
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
siamese_2/scala4/cond/switch_tIdentitysiamese_2/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese_2/scala4/cond/Switch_1Switch siamese_2/scala4/moments/Squeezesiamese_2/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala4/moments/Squeeze*"
_output_shapes
:�:�
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
siamese_2/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_2/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
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
 siamese_2/scala4/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
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
siamese_2/scala4/batchnorm/mulMul siamese_2/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_2/scala4/batchnorm/mul_1Mulsiamese_2/scala4/Addsiamese_2/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
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
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
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
siamese_2/scala5/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_2/scala5/concatConcatV2siamese_2/scala5/Conv2Dsiamese_2/scala5/Conv2D_1siamese_2/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
�
ConstConst*��
value��B���"��T���`����<,eʽ̚&�(���_�=ą3�0k�=HDZ=���?e����l���ٳ����;�ā�P*	<M@�=��/=�$���`;��,=��J�T����?<<$߼�3'=�ߒ��,O�Da��|]k=�T���X���<��<l�ͼ¶�=��V=^4�=�T�HT<Z]���M&<$0�=�?�<hg�� =���<82��T�� �N;�Sd=�˽�xH�=*QN=�?o��H�,��O'�=7���Z�<��=�0=X�<e�
ȍ�+r<J�J�P��;!ҹ�A�<��=��ȼ4 =v:R=�X.�����vJ=ЛS�8W��"׽��=l��<R���i�Оv=�^̽rւ���"�@g^<h�p�`>�<H��=\:��d����l�.%5����$r��K����r�=�~=r7�xϼ]͈=8���P�=������=�&�<m��P��=�&���=@��;��a;r>2�vw9=��=�b�Ԣ�=ҏ�B >���>(y�<�M=P����̻4U0�0�,<�ue=������h�n<���M�&=�5�VY�@v�;xkM<x�q��r�<�]��b��4
ļ-+� �<�$Ļ �H<�y��Օ����r=��F��%:=��»���0<t�0�\��~����0��*�;�:���(<�u�; Pd�d����Z�<�1=@�B<X�A<�Xf�`����m������5���::П(<��ڻ �;���<�𼸪Q= ����ŻLM��D����,��pR�< �:���؏`<��_<����"K�X-㼌�I���８��<}�'=@��� �ڻp[L<h5<d��<�X���9ͻP�<���;���<��м�5	��BԻvG/��5�������)�(�_��	C�B+-� ��; �мȢ���$a�ZՖ���9���p���GD=�"�<�ර������Ӽ(�I<�=����m�=���L�˼pu�;²4�������� ��;�^r���*���3�pd�D��8�D��<���<��y� ��Ks<qz=������7�:���hS�=`�߻��=^(�=y^<�Z��8���3��|���`�;
v�X�<�X=l��<JK����t��+�=��!��r�� 4��. =���<��:<�G�@�<�-=@���(��x��@VK��7��>~=l�=�Ų=�i|;���<4d��(!�<g�}=��=�I�:�.=6yx=A�x=�=��=�3�=����ڳ=	(r=���< �8�^�;\T�= P� 'y:�d\=���<�_|=׻νL��<�<$ۻ��S<��m� �&=C�< 5��(Td=�b�<`�<2�w��E=��*�����`��j$=xM�< l�;�	�<,�=����7;У�����<Ys�� �9=�=����	t��{[�䷸� �9��:�3��8����a=��������m=(�P<��B��O�;N��=�N=8�¼]�=����6=��==��<(׼ϔ�=8��<P��<KD�=��`'c;���=@�;�ê=إ�p8�<�ш�����P~ƻ�n��L3���:�<���5�>=�P���+���(�����L5����X;�	�A,� �=�;�ܼx�R���3༤a�< k�8�<P ܻ@��<���<5�`�z��·��,��gD<�kC�`/��8v5���<���E��b]�`(����;\Ƽ(������; @��pl��'Z�w���S�ȗZ���<@�L��"���0U=X�ɼ�c5=����Z=(U�<fI���ɻ0V�;(�&�:< 
\;'�#=�^�pA� �`�D�`�NǼpC����:=�P_�l�д,<@ٳ�9�<W��]�<|'�<��	<�L�;V���׾� ��j�9���.��>��3Ǽ �k����:�	ټH)���qȻ@��� �.7�ぽ������� 6�9��<`��<�ؼPd'���p�;�j=@*t<H��=����Ӣ;p؁;�uż�I��Rü ���X�ݼ��  �� ��90ñ��,�<��<���IW=dI�<��F= f���
�;H��H���6=}�A=��=H��<�؟����=\���T9��T����m�<�%}<(0�$�K���S=8-
�!�$=p$���=�~���=8����m	=8p�<8f=�n{��>�0ŀ��� �೭<���=�F�=�Æ=(!=f�D��<��)<�c= n�;���<���<��
>Q��=C=jR�=r	=�=Q'#=h")< Q<0�=�su=�x]��o��F<Q�<RI�=K�Ͻ"����:A< <|<�ŏ<�2}���==�¿� �E�= ��N=�哼�$=P��<��<��K�Hi�� 뼈�<�"=��.=� �Cs^=ԁz� b?�8���I�=��&;�>�<@��;�h=H��<��l= ��<4�d����;�g1=l��� ������<��=`7'���;ʟ�=��`����<��=`Y<H;�<�r�=�:<� ���e=b��A�6=��t=��h��<�����;����=�x%��L=�y�<\~������`&��R����=е���3=pc���ۼ����Ϫ��)�X"q�d| ����:@��=��<�%�^����|� �:~�o=��ڻ@��������<���:t���y9��c�������=$�C�lS�@ri�l��<��*�V�'�Ѓ�����h�缚�"���Y�<�Wh��X�;���@�z�nP$�����Ђ�;��D���{��4�=��<:V@=��d�fz=�=V3(���<@���P^�;4��<��˻��A=�����< �f�����vĻ�t��V�K=��м��˼ ���}��\��<�m��s�.=@�;@Ԩ���)<�u*�`�����;�ۼ�8K�D��<𲢼�RF�5<X�b��0�@��;T⥼|9=���@�:x�Q�|��<�����.�<`�ʼ e�:��� ��:�Bt=�:�<(O=����<0�	<��L��+|<PA��.����<@_<��<p��;`k;�x�%�=��=,���$�f=����h<�V�8c0���ٽ��ǽ��o= *r;|P�=ৱ;(<�Y����==�=�][���< �<����I��(6\<��F���޺�<4�^ �=(�g�oܨ=꺓��=@n亀�úȣ�<���=�N��H�N<0��;8�T�p&H���4=={2=���=���< �<;t|�<A޼��<HN<@~�ɶ�g��=M��=)�=H�#=��=𹷼��C<��H�J�o�H=�6=���@c!����8C����X=9~��,lo�Б�;P]�<T��< D�:ʫ =��}�h'<�r&�`�����:<�{q;���<��
=��e=�ڟ�f�n��s��%<�R�<X����;춈=Z��v��6A�19]=����xh�<�=}jt=��G=/&�=s�:=�0�H�k���!=������<8W
<��2=(h�@���t��<����U=7�=@徺@b3<��=����\f���z:�3���{H=@���<��V;����e��r&<(E�Q�=�{�<L�����ּ�W�<����^�<�l.��L�< �~��3�Ԟ�d�Ǽ����xiL��&��=Z�=@��:��8�����0������;RS�=��E;P�O<�c���a껀�y< F�@�ջ�ч�@�(�t��<�W�����^�:�#=�é��d༈�6���D������;��c˼�X&=��:�l<pc��@&��Ӷ��dT����; �:��`�	�=��<��X=����tԸ<�O;=K��\=�ϻ`�y;���<����"=<����=��C���D��JI���k�b�Y=�t����еM�H2(�b�;��&�e=� ����:��#<��h<@�;��;����6L���=�j����;ؕK< ;������a<���}=��&; �d�PC�<4�/=@��xv�<0���� �;�@�h�U<��3= >=РL<0��G=\d�<�>��=@�<.j �Ք=���<\�0=�)�p��|��b'?=�Z�<�ݴ��Z�<�cļ�ؼ�O�(�i� �V�n��4�< �;��= r(����;�8��0��<���=˩���S=��<@�2<>hH��� =2l��J� �T�$��=�����=ܠ�5Y<8�;�_ێ��1X=F3n= �����6=�Ee=�C �����P	Ҽo�F=<��=������<�|�<�� ��������<`Ѽ���0+�;�=��=���q��=�H� ߀��Н�@,�ιC=\�(=nq�ܸ�����#_�|�<�� ��x��k[<j4=`=t�����л��}���;�����A<�"�0�Q<�\�<�(�G�< �7��B'�����	; &��HC˽�tj<|h�<]$�����N���P�<����`�<���;�9׺lX�<f0@=��s<< ���D�,�<����	=��=��=�U�@�m<4+��M��L�j=���*�� d�;Rx�= 3�_ڊ�O��&�y�� w=x�ۼe(5=Y��@�m� �T�\�L�x�	� �9�}l�Ȗ���N*�t��<�����"]< Q�����<@�6��� ;��j�,2ݼ<���d������=r70=d�ϼ��9�J��`
$��=��Ai'=�Qb<��=���r�� ��<04� ��:nԼ�d�p���L��0����1�Y=�nļ`<��pË��랻�?H���>���+� ��<`tx�0�j<(ݼ�+�� ���!p���; է;E�<�I=dT�<��i=�������;"&=Ec����< ;������P"�P�Y�h'�<T��&o<`��x�#�T�#�h4x�m�x=`���ʻ>��V��`�h<���:,gz��;=8{��h![<H� *R9 �:��軆�
���I�<�vL��;@��;�+!<"v��@��:\���0= �: ��tE=%+=̊����<�+�;�,����)����<���<��.=p��; -v;�
=��<l�ּd�<\S�<Y� K�`s�<x��<�L"�~��s7���b=،.��]��ѐ��
�#G��L0��࣡�hhݼ��ݽ�����5:���<l�ݼ��ռFN'�d���0�H=����h�= j�����<��ļTN�< �8�X���h0=�nF<�ֻ�xK=�0A�|:��*N#����8�3=\z
=$��.@==��=�r
���x��r���'=��L=v��Pq<_C<J���zP��j_<~�VV��
u����<��Q=�ML����<�򏽸�0_黀pU���=�]=�Ľ�Ƽ�4�ĝ���`��:�< ���,;}l=E<b�*�\r��x��p邻t=g��ݨ<<O�@-�:���<�k4�d�;�����ܼ`�������^�������8�輬ͼ'�Ƚ�V�;P�/<Fn��ܽ�^@��8��&�����>��P&D�\.���?�<��.����<5B=�;x V���y<�)����6���T=�E�����x{]�jQy= �z�k7��!5ŽvW4�Wde=�����<*@)�HY㼰�;n�Ľ $r:Ƶ^�X����5���6��,�<�G!�8�r�� T�9��ݼh�<��/<�]� ���{���^9���6<l��<V�L�CΜ��d�8����}���_�8<D/e=�w��d��<�����,<!��m�4�f����t���5���=^�_��o���ڝ;PP}���?<陏����� A)�x�>�8�<�K��-�0�<�¸��@��:'�"=<�輀[S;�7=�Շ��ڋ�=M7����<�2���d�8�_����p�=<��ʼ��@�`����1�vr���Z��]=P��;{C��d��`t=0}��%ٽ��3<*$��=.���P;��p��(���2{�
�8��qX���6���[��s���#�<��ν�s������|{�<Xpz�,N����<�q�<���x%�<������ļ탨��= �{:iqa=��4��<�A�<��z<r�O��< +;�4D�2��̣ͼP��.<x�\©��ᖽ�5=�,I�J2S��E��P�M< ���,ڽ����@�����=�,���>�= m=���}6��ri�(��F�;�~7������=��=hh�<��@o���W=�.���m<�\�:�W�;���<�5<Rvl�޳~���=�|���S=lڏ=��R<�8Z=��F����=d�M��D�<������<��=�����+�T8�<��׼����p��<�U=�&�<����_�3=�]=P�
��˦;���N�>Iڽ)�f=4��<�đ�D��<Cꬽ<����8=�WQ��̹<�0s��4���1<�W��I�=\q�=d]d�I����B=��f��8����Zđ=��|<�<�Ƥ+��X#��YŽ0A��@N�����;�Z���%=��=��ż��½�3��^�I���(���r�RMǽu.=���~n<�8�=���� <N;Px<d�=�׏;�	���r=�Z�x�a=(@�<�^�:�.a�0��<�*�<z�;�Vy=��<����1�=�dW<�:�;�Y��h��<htǼ,��<�r/=`D���1����<�\����=4似N������ ��; ��7p��84&��V��z��F�`Sg<tJ��,r�<케+6;����([<`5�s1=�^�h#O��Uջ@��:I���< _Լ�M< @Ը��;PՓ;�!���y��P��<hU"<@�!<�) ;��;@Pa;��лR�N�H�0�`���p�;�|�;�<�<���c=Ȑ��P�4=0Z��~Ի�*��T���I�@}R<E�<��;���<��@<Ă�����`�u&� 5�̏�<^�0=P�� 3<@�ź��<5= 2�<ے<���;�V���<�-u<��̼`�ڻ�����ȼP�����Y��RK; d���%�hԝ<`�M�H<주�y����HW��3/��~�<(Ck< 0�:��غ����͆;_�=0~S�#�[=pj��p �� 0�9b�G�H��� �z���b<�I������ j�;�D[;�k^;0v��6=�=�!����;xI�<X��<�벽���l�b��L�=�����=�~n=��i:��0�4� �K;j/<�"fR��ȣ�Y7=�d�=�?޻d����&��d�=�@w��!=�̫���$=H'�<��<�L>�@َ�O�=���X�g�l��<�
=���<��L=P�<�n�=Й��m�-=
C��䕻<h�i=�s ��@��6c=���;���<��=7l�=�X=r� �k�T=MN=Ht<
�=h�<�>6�����<�=��<�p�=����
�?�&=�ޣ<8p�<��F���2< ����Q<<�=G=��m=����yω���^=h�������wĽ�p= ,<��<�.u;��<����8���f��L�<���z�_=ڝh=�����4�8���,)����=��g<���j{��{=TӼX򊼪>�=tɀ<����`F=��|=��<h�<s?�=\=�UD=��t=�+�@na�P%=��J<�=�(�=@��< �g��l= ��:"J=�T�v�r= ��< ���8w��ao��%'���9=�p����<8��
�@�F3J��� ��J�*�L5�PxI��>= �������L/�lV�<�A_<Ɂ =���6�`tZ��=�G���iR�\�H_%�T�ټjR=@x�������Bǻ���;`��Z�/��3Ǽ��;��������:�#�L��<�m�:� ��;-� L2���6�`DC����;�a&�R�"�!��=�r˼��<��5���=�<�./��4� �S9)�=�|�<���;��=Է���<`��<�o<��<@�8���=v�C���ݺ�l�:����l�	=�_":zo=(N}<��ȼ�s<��u<�]<��
�;�h��QT���<P_���3;J:<�.���<�
M;�!<��.=Hmp� Q��ؑ�ȗf<�\<@��:�� "<`��;8X��C�j=��K�WQ=h�8�<�ݻjG���b:85���T�0��<��L< �<�9�<�n=�M�G=;�H= �p�Ň=��1;��;أ��`���Zڽ���@��<��1=��= ���3¼D��`�<�쀼�R������^M=��<�Nx���J���U�=`8鼗��=���I�y=�0;܏�<�h�LY�<��<=�!=V��hR�<%�X�Y<l��<B!=��=�[=5�=�dl�0Nd<l'�<`�0� �y����< A�9F�=4�=���=K1=u�+=���<ty=�	<	�<P="�= �p��5����;Ġ�<zE�=���P�3���=�}=@ԭ<��%����<Ƙ���#=��< ��`C�;i#����]=���<x>H<����n��p�鼂#=h<T<��<:,�@��:��W�ܟ<���"�=�y�;�G��/V=&�=�=<~A�=��4=��w��2��&=��?� �`� K�<� =���� 0�<ߨ�=D�<�d=�=�3$�hT1<�h�= �ܻ0|�;('�<��q�6=(JK=H\= �Q�P�w� q�:��b= aD��Q�=�&�=�0v���A�Nꏽ�0��O�n=T�ļ��<���:�hc�7#��T���x�kP���r�<T�<dV�=�0û`4;��~�v>=m�_=�=�n�㌎� �;q�=����`�<Rkq� tƻ������=Xzc�x��������W� ��;D��vf��k;$f���7�B�r��1=�BG�+�l����<���8V�`v�;PH���9���X>0w��<�}����2=(�U<�4� �u�������y=�M8=0o�����<xt����=>XY=��Z�!'=ؿӼ� =Ͻ��\�;@mѻ��.�Ц�<H�<0X�=���<�t���<�?=�+�;( x<���;I��@=p/߻(x<���<L��,�<p֨<��<��= ���=ȣA�Ac=�+g�8����(�(°<�^<�����p=<Ϫ����< W��)�W=��L��������<����C�����=�I==��<.A=��= (���Y=��=�}j��7^=�'׼����3�te��l[�P����3K= &�9(=�=8Q�^g�Q/��l�<�O<����i,<�+�< �;�5���ʼ�&��z�< 	�zֱ=,����̤=*-�@������ ���[<;��=�a߼"%=��w�@胻 �W;�c�<|�=���=���=��S�p �� c�;x�V��0s�PR3�d��T5�=��=���=�<o��=����%F<��P�8�� �=��=�Ȼ�˧��g��@G�;:��=�?����m��N�<��(=�=^'����<1ܗ�(�.=�sv�`��0�;e���M9N= �F<��7=�W�I����O:���=`�u��ܼ`i��0(><4]��h��h�)�:�H=��4� ���BȌ=�*=�=9�=��q=��x�C��$N=�j�� T ;�` <G	=�0� �:�K=�?��*�=jC=��뼌^�.~�=x弰�P���r�魙��V#=��D�̉=h��ȓ����<��<8�����=o�=���ȚS�>Ǧ������Re=&���;ֻ��b���a�m콬���56��}��◐��bu=TÔ=XX���}�<a$���$=e}�=H��=0ð��姽�W�;�<H�]��<=)�����;�N<��1�= �#��/׼`����ڼ��-;A؋��\Q� D�;�D���1�֩T�4�b=�{>��R�����`��<?���P�@K��A�喦�?�%>0�X< ]
��l��K�=���;�G�pF�;XD�Om�=�!i=��ۼ���;pY�祯="�=0jW�i�P=�ؼ�s�<3�ɽ0�)<l���gc� �g<�]�<�o�=���<�8��D �<
Ռ=tY�<�ڠ<�s�;���mb+=���x��<P�&< ���<h�<�ց<;�=t��=�E���!=\�0��EH=���(�"������<��#<�oM�I�&= �� '��`��;֓z=�%������<�C<ȷּ�r
>���=T4�<l�<qс=��}�=��_=�ʒ��h
=�0-� Ţ�� üh��,�i�+��(=�<�=�q���M�Ǐ��xv<�>�<4���v�< �`:�A<�{'��#��Ƙ �p�ɼ �;��P=ĸ#��~�=Rɒ�4��N7'���ý�l�<V�~=�5��2<]=���<5��B�Pr�d�a=Q�=hE=0WL� ��� }S��Z��`����.��F��P�1<�=�K�=`����ƴ=P�&�p6���컸e'�giz=�<�;\�������F<�k��y�<^�0��`_��'=nЊ=��=Z'K��<�����<8����'��w��l�E� =� ��"=�߼��|�0���t<��мX�g� �ι !лx�}�R�2���� �L<N-Y���s<@$�<p̄�x.�<,=N=���;�c���]=����l��<�9�<�c<ި��pԮ;�::<g+��<��=x�����/���<�y0�=t�s��,�����#젽�=X�r�
=�8N�Ƴ��	=�tT�`�ڻ��=`d0=�c缔�=����e��E�=�O�a���2���?�н4�=�`fe�E-�����j\�=lZS=�C?��Y<���5�<�x =f�=p��J*v����>���<�\�<q*��࣋��4�m��=���:tk�������K� a`�Bh���9����:�=���5.��^���U=�k��xr7�4���]X;66N��9����\9���ek����=���<�Dw: ��C�<���;�\��;�;�E�G>=�\=N�-��?���s����=@ =��)� n�<��鼀ͼ<�e������ ���� 1����ջV&�= f��^I����b<e<I=���< ��;8����ڗ���<���1|< =��㻐����;�(�<�J�=����U/<�j���$T=�!&�l&��f޼�h<�湼�)���<X�ż@�D�P*�<|�i=������Q�<�c�<fG�W�=*�=h�~< ����h�<�	r��y�=xE�<�Y��'�<TiC�b9�`���D��HV.�b���Ϛ<�F<�}�<ģ���S��(��`Cb�PF�;4��p�<X���aP<�߻p��;�:μ�1���=Pw��S����=����y���q�N�*�$��<��="��0=�^o=�?*��B ��z�D�<A��=pD(<��k�Ht����޷G� �&��D����������z=��=��S���i=������𯬻8"ӼfnQ=$k�������}?����<��C��p����9<�n�b�=�]�=��d<V�f���������x��Tj1�@v;��ּ�/�h�z<�G��Xu�<��4�j�7��"q<hܰ��?]����0dӻܳ!��x;�ᠧ��I:��Z���t�P��<.�H��@r� Q��ؐ��xI/�$w�<H!i��w$=�y ��w~<؟�<8$T��Ƚ ~�;­�A}����`=꥽P!W�b;z�n��=�����Q���榽!,��d��<6�����<����0��Ru=��߽`,[;t��<`�p��\0� �)���^���� u�;�^�$̊����ԏ��j}q��FE�8����et��hV�GN=��=M��ƹ ����ػ����R����< ��8� �����)R��Oj;H"k��D:�(мH4� �:@�U��zw���7����<Zp�vK�|V����I�$!�+���,f����<T���p����@����dڼp&��w��d¼�1��P�g=��H<�ZٺW����<h<��0J�;�������:�[����u�a�DG���=t�ּ��.�@
�d@*��ն<R"v��ۂ���3���<��"˼XFv�� :=`�μ�=�Y����&<H�c�������j�����$�z�F�����દ� ;L�g�Ժ���V��[�=��(:�����/���=�p9�`����Ҽ�� ��π�x��@yĺ��;L0���<�O=��;Z�u�P<*<lJ�<��=�� =Hi=������-�DG�\P%��u=��|�W]�$70� �㻒7��j��ztʽ�Q
=��=�����>f�=X�:��_c��y����<<՜�Ư����<���=a��=D �<ꍽ���8\��'���#�<X�<H$�<P!?�j'=�Qj���:���>@t_�<��<\�=B��=�Q=e�-=ڲ����=�n�� -�;��h��G<B��=�9{��L��V8=�����x��6�=r�=H��Cs߽��K�C=a=`�h��>W�@И�]�><r���5=$9<�Z��oWr=�p������0 =@.;vO�=�)<�0���x�<�X�<��<b��=�l��'�����H<�����,b�Q���:L�=0=X�k<����P�\��� ������e <|ٲ�ܚ��`n<�;=�>�Ĭ��m��_Բ��1�;(u|�������P\=v�.��,=�v�=��_�@���u�=��<\��<BZ�@a��ތ�KFt=���<�l�pkм�켼�^�<��ɻ��2=�2�<�_x�dڢ=����	n�xDg<N=X ����<�v-=������Y��_='� � <|.ϼ�$㼍Y���*���X�;ƕ�|)��(*:�a!��J���R=�D����g=h�	��=�E� 6N��a��TG�<@�ݺX�.�b~ �,r	=P��teL=P��;�n�<@�/��;:�P�X< ;^�X���K=��-��Ud<�<؍=�LJ<@�g��m��ջl���1�0�H�|�փ7� :�=ȝ�<�P�;`���.� �Ѽ�lV���7<��<h�u=���<��ټH��R&?��9=��#=x�S�H��<Tv�<��1=1�½���4v�� ��;`qI=(�<<Ƥ=��J�4y)����=��=(��Hѽ<0�������Z�<�q��bp= �D<^W_�b�=��3��D>=��X<p��xk���;�0�4<P9�<�p"5��
"=@g:;pn�L��<(=����< �q��[�<́��O�����*��߿<Ȍ�AC=\T�=���<�m�<��<����$��=��=qT�0�� M���Ǽ�F����&��s �R��=h�ֽ���=�g�=�$�H�����<�&�;�ݼ+k�ЌZ<R˰=�ݑ=@���Ja����o���<�<d�,�B=0'��]<ܕ�L�=��W�>�=��=�f>���=��< �<%��=��
=��U�^�^=\ q��,�<�L� 帹�7=��H<��Zh=��;�f$����=z�U= 1��ݑ��|��<���=  +���L< w�<Mn>�T�� ��;XU�<�ַ�#��=�����Q'���<@k;��U=�qC�`�W��e<M/M=� �<�ҏ=�5Q��b�4��<�ɗ����᤽�"=t��<R0\=-<���:U�Ƚ��9���2����<�~��xv< '�<�d��Da��n�j�*����=t)<�c�2b��D2�<��*����n��= ��:P&E<���=R�=�b!=��4�&�<�/��⟀=�n�<�E���=X/<j-= ��9>݆=�0=��6���A=�׀��&==(qL�O`=02}<�4�;0��;c��%���Æ=���x�
<U���%�s9ǽ� Z���\��'N���O��-<�A�<G��C=a��,�\=���<�w=]�Tc����?��V�<�q�:��p���F�X9�<�{��V`�=��<� ;��1;Ѓ���{�<��Լ�Q���*=�Y�ϻ�M]���2=�\�;����2�]�p��;��X�@��Lr��T��K<����>���<����I�����<xZ<��?h��Xm<�<��=rN%=.���f���1��ӫ=8-i=�j5�,��<�@<��=����@e��8��P��`�8=x�z<�ج= ����j)��3Z=�-{=��5;�!7=���A~�� Z�<�+ɼ�i={&<�'��$!=�Jۼ�
$=�C�= �@<�3��,f�<�x5<��%�̦!���Q=���;8�9����<��?� �)=X�O<�hW=����၇��9���f=�H��yK�=�s=Lb�<��=��p=�	W�b"�=Ӌ/=�kӻ0#��$�$���ڻ��M�4�<�N��H=� A�x5ɼ�S=x�� 4�:(��?1<\5¼�;I�X��<ND=���<WY�����@6;��=�N�|��=�ΰ��d��pΧ<�<:|��~�=Ï;��O��L�<P-�<,�缪�_=�n<����.�<���^��=�>����w�<�Q��
��vNY=��;�<\j�=ʁ=h�D�pQ!���Z=��=t������:=��=ȩ��lk7��ƻh�H=��=����HJ����7��6м� �<�I��Ā�8a��Ę�=p��;�`<�8�#2��ȍ�<l��<p1��(���{V�J��P�k=P�;�`Q=�>?��/��M��v_=_e��@0�;�덼��^*�=h	�쏘���o=+�0=�=�~셽�W��oM���U����</�2=`i=lj�<�V=��;��[�Z3m=��e��:�<@ô�����h�=t��< ����\u�<>7=��ػ'�� p�:/(�=�%0�mU�=�.= r(��fj�>�ǽ���@��=N��`����^�6G������G��@���oy��1�0=�~*=��y�R�=�2�����=�l=6=�=���烽`` ����<��;y�<�놽\j=_�^�=(%�<�֙�ж�;,,ü��<�#?���u�X�=4�����ܻ�뷻D;{=`Su;X��: ���<�b������-��r�'��ǽ>><>RjF=�#9�y��h�=\���Z,���<0�b<!��=أi=��T�P_μh׼f'�=��=<�3Ef=��;L��<x����;`5c��,�x��<
=��=xWi<������a=��=��<w�g=�d.���ý%�<������&=�<`P˼ea=IC��s=V+�=��I��v*=�/�w�= ��;��}�th]��t=0t<�������<�抽��=�8=`o�=�Ӽw����{/=�K���C>��=��<�8=ʓ�=`�;���=�c=�I���<�%g�@0��淼� �<��<��E�� l��	��xB=�ϼ<s��H[�`U�;.�M<��8�2<䜢<Џ�;k���pv� ~z�<�s��= 7Ž�	W� ��:6|�
��EM=�_�l��<H�L�$+�<�1����<���HB< ���0(���T�=з����~�P�F<*��$��� �l<�ꗼО�<Lޟ=V@=���]`=$=@���xdڼ�p��Hi=fZ��B<V����
�tԆ=*��=�Fi�D�%�@����ּ��<��_��c��6�`� ^�=LW��d���ϼ�Hս$?�<�9�<P4�������ň�$)��P��<�~"��=L���T0����J�=+D����0˼�ba�r/�=��;�v<�GA=-HR=��Ҽ,�Q�T1<meʽ��V����;t��<8eZ<|��߹^=�!���?�O�=�\ּ&�&���h��o$��7a=�h<��3��t���Q�\H�<z�$�]G�l�&=s[S=Xb�T'�=�G= �H�dkݼ����t�=*�i�H��|��6�O��j'��� ��t�:A葽V?���)�=���<�����U�=�Q��9��=�ݘ=h��=���r���3:���<��<��4=���{U=8O9����=�=@��:��i:�J�m�<�W}�Wh��8z�<�|ƽ0�c�`m�:�=����D�(�z�� ��<A*ŽNA�����j��i�s�L>j�j=�������<�
����]z< �<_X�=b�~=����
 1��zs���=���= ��b*�= V��F�;��,��qB<pQ�4~\��(<99%=3->��g<p��z*�=f�=l��<x~a=�#5�MԽl��<$Pټ�H.= �O9`����j=��¼ę�=��	>@g;�?=�X��?7=�!�B�����c���p=��H< U�����:�ǯ��]0<8��=�˛=�����}�x1���J=�Ƽ��8>�=�>�<
*=�
�= o-�?�>!�S=l͵��o�<l@_� �;p6�;X�:� l���揽`vG<$�O���=豳�%	��Ey�� m0:B�<�_������H�<<��;��׼,H��Bֻ�v��0�;��x<<����w<Բ��ժ��Y)�*�,��Y�<�sʼ4ť<�2��0��@I;@�[�`^\;��=3�=�3ý±�� �o<�:0�0ƹ��U�,�0��/����-=�d�=D�	���=@!�|!������2���e=/���0(ӻ ����3�<v�=�G5=�)�(m�D��<PH�;�t
=
Sm�_��Ub��
0�=�@�8�μ��F�՞佘��<���/=<�������z<�2F��k�$��<ht;��*@�I���]?��ֵ�>r7���M��%�<�Y=��� ��; �9�@=��<T0��Z,=�Ÿ� .;@��:����(+��.�@�'=�}��҂;��y<�u:�%�����<蛽`��<Bu������)��뜽��<w������=,� ��}��n�=�O�<t����#�������N1="#n�l�N����0���b��`$��m<�x�G���Ŝ=,}<!9߽!��=�\��o�O=�%j=3ߚ=ȏ���X�`� ��d����=�=���/�4=� �Bc�=fb=D����F���� � G�9��J��͂��]F<��u�*t� �Ǻ�)�=�j0��좼��4����;�_���j��1%�i���j��"�>h�d=��h�2����'=���� �@~K<�K�;\��=�=;.ý��\�����=�C�=���ȱ�<?���h�4��0�ռxzؼ��PV����:���=p������Pqa=s17=�U�<��=����@�ǻV�>��E�<p��%����<`*��X=�o> 2<H.3<����	�K=<�������t5Q��B1=l)��J�� ����I��&����=�֗=��A��m��'���%/=�_��>	!�= ;���A2<�~.=������=|h�<�V���;��y�@��;��r<h��֖=�^̽�)�<`���p?�<D��Uh������(�"��]�W���m*�0ϐ;�a;0��;H�0�H�`�H���H��<L?��<�n�<��_�}ؽj�\��y߽Xc�gI<Xζ� ���@a�;�خ�������X�\��pm= o=T^�~����R<�2��䮼cԈ��uY���`� O�;a�=b�Uc�=��!�8��m��Rǽ+_=/Qý\�	�6ϓ�R�S=@�<@��:�`}��j��E�<f�<�C�<˄����������Wc=YH�<�Ҽ����ڽ�:!;x����< Р�닽��=����`�;8"\�T c����P�2Cg�/ނ��茻7�:=(fA�a�`�>�W �@�<<x
#=T|�#b/=����x1��į��iD�i���}��(�T<f� �{���`-�<����l)�<vjĽ𧑻ZqW��A��R�\�ҽpW����ݽj�V��=�1����{<�2�<P���Z�,�p�3�>����J���Y<�qh�����0� ���c�ƽ�#6��5�;�\�Xp�D��= �(<��Ы�;�A���L�:T�<�L,=�S�b���+������=�K<Vq����<�a"�=@<$&�<d�[��5/� ����x˼j��@*�'�����m���-��g6= _"� ����_�x"�$��� �r�8�2�f��v3�Q�=�� =��O�ޢ�B�=p��vI���x<���؊�<���A׽ �߼zZ�E�=���:�ռ ��\��ٮ:wA�!�����V�(�Լ�$Ｄ�J����=H�Ԇe����<`F�<�� 1�:CH��<� Z��1ד�/<��>�q<�<��iM�@���rT�=���<����D�H�4=��"�(��|�9����<�	H�(@�������@���޼B2�=+p�= C�:Џ���ļkS=�w!�v��=��=0�Ƽ���r� ��^�=0�!��+8�hР�XL#�2����u�T�j��l�<�.�=������=R��=��ƼX鄼1�= ��`eĻf�2� �;*F�= �=f#=Ǆ� ����M���ϼ �j:@���`~^��d�>=��<����<���=�`���e=�3û��O��=�,�<<,��p�V<r�콒��$��<��`�N=@��:;�����=���<���k=@%�<p�W�'�Н�;�j�=�%� ������P�>�(ٽD�<��?�T�ŽsO�=�a �`�%��*�H���=:jR���;0pH=p(=@`:;&�=Llf�eˑ�z*��8t��!�b�i�9��=��<�L,=�4�<��h�t���(_� ��;�:f�����-�0����R���UD��&���۽ �	� E��,���Ͻa=Rjj�HV�'= 64��e�<��=�kK<��4={��� ⼂-��d�<= p��&��`��;�m».J�=�1#���k=p&�;�"��Gw=8��pF�<|�<h)<��:���<�.=���N�_��~k=z����:�ژ�������Q������؆(��|J<�c���I��vW=�Iҽ��S=pu���%I=R�A��л�0��� �G<x�+<�k*��`���R=DX�1�Y=�j<��#= �'�LԼ��<<q�<�J��8a`=|!���W<�q=j�=(��<����E[�(/�г������W�t0����=����=��J=Pٔ��߰� ����f��jpd��R�<�A����t=$��<�4:�N�9�z	f�'!�=��A=8Bq�e�B=N�<$j�<�V�0O�$]�����#nz=p��<C��=,����'��e�=���=�r̺�=`�+�����y�0=�ˠ�s�$=p{�<-ႽpǾ<l����a=�ƒ;@���	����; �?<P��;زJ�x�<� �q=�`R<�����O�;LU���"� T/:k�7=���}��$�J�&=t���/�=��=L�=r�=�ħ<�r����>��=�m�VC���}e��D��~��T<�0���>�����<h�=n� �<qg=�x8�Z�� J��{8��g�=�)<=x<�ƽP��;p'h<����<S�<����q�<�<��=�W7��5>P�<P������=T�ͼ�������=x�<��d����:��� �E���;<p\�L>=У<�UE�-��=T��=�=���=и5���&����(��=�ڮ=�ɼ� �һX	��{/�=������C�p��;(�����=��� O���1=��ٽ��s%=����pu�Fv=��<L�;c>=�C���s�Ht�;�������d�=�w�;8��=H��<��=e@� @A���Ҽ�gv=b*�.�#�����'��2�<�~���e��hQJ<8�<@��Bj��p�ּ���
���K�<���<D�r=�n�=�=V�=�;����k< ���x��=��z��T�·=�}1=x��=��}�N��=�3�\�<��=p1��B�=�9��i�;�� � X<@��<�aF��ڍ��~=,,�����:P���h�>��P���'��p����lG��)<֍��SS�mg=O���p�1=�SZ�CL=҉�@3������ЍN<��t<�^ּ:|��d#=�岼$�6= z�<���<��M<@-�h�=@˷:�BM��1j=�?�@�:��=�Z=��< o����7���;�K⼨����"����E�p,�=BSu=b@�M���-<0�o���Q��g�<@��:��B=x��<B�P�F���l9�*ֹ=�k4=���=|=���<Ӳ߽[Ǽp����T3;	(U=�x<ī�=�v��$s���si=rY�=po�;M�=D4��錥�8V�<>5���?=0 <.�!�h]�<tL��q"=�*a=�f�`��; (9 N�<�<8������JŇ=��W<�����;�"���<8k�<�n=̷��ߕ��g��j=�rĖ=P�p=X��<0S�<��==�+�;RO>$��<X�< �^�)���B�<<,o����=`��#
>8Ƚ7��0�=<���ğf=h!<�}���m�%c=Ј�h�n<��&;6�x�4�P��<��<X�)���:=�Y����O�a=���;�֢���U>B?���r���**=�E�a�����=X������:�㹽̨���h�<�������x��<�l���KԼ�W�=�l=�+<��=L���<=ټ�~:�E�=��= ���7���һ�Y�(��<�j?�ئ���W�=��=C��7�e�����P�"�Kɚ����}�=��=��Y��tǻ4ql������\���=��#д� C2�ȿ�#=����>f�m�X�R��h���H�=='��	-����,��~^�3�=P�ռΖ!�d=$�<l�_���=�2;����Ƚ�˝�����=�r=���=��.�T�=E�B=���*��=�dJ�[�*=��p;��,��=��=�^.=/D���s<��.�<�<`�j��2>�I����<�{�:��e<�L�< .g�ˬ� �i=8���ቼ��0W��]G�� [��Ă��|���q�=�=�2	<)���0�=O���+w= n�<t=lS��m����P��;@k�<�ր;~��Q�D=Hg���7_=1=`�<���<�����Z=�b�3N���z=x�=����;�8=��@=�UC<�����{༔s�<NH��ļA���#@�[҄��K>�=�s�����j)<D�����޹ =���<
��=��<'�����E�����{�=�g=�	[��6=H�=0��<C`���I`���<�J ����<��<ʞ�= �޻(�<���^=-�=�KH<ġ�=�#��!����Mh<�W+���^= a���z�����< ��(YN=*��=mQ<�== ���<�/�<x
H��v��=��<�u ��n'���h����<��=^��=lӈ�q�����
��3�=�C�����="%�=�U�<褱<��=@�<�>P�<��;��ʒ�����<�����
�=��!�'G=Lht���:��<v;.��D�<֟3�d���u`��x�=��z��/;��+o�u����ʽ�q=s<�]:�Գ�<��ὶ���/=j#&���f�:�/>Pe޽0:E�pbջ�������f�
=P��\��<�1ý<X̽��Z=�n���ѽ`V<4k��;���JK=���<��;��<BM��LD¼���<��=x c�8{-���0;��	��0v=^˓��s#��V>�=Sլ��%e���������#���@�W�p���Y��= �����pC�3F� VN:��=�ܽ�⩽Z~��T���<�
��Y�>HBr��x`�wڽ`A�=q亽�۟�d�缶�#��>pȏ���n�$S�<��<L�<���������W׽6=[����T��<P�=�߽Iߊ= 
6;��ȃ�=0���t�޼��hV���=:�u=�����t̽0���<��|v��{���A=F�>�W��4=@��|�<Hp<6�ѽ6��V'=F�>��>P�\kg����	���O���D�X����瞽%m=dו������=�X��dN�=$�K=V�d=��ڼBl�@���B6����<��<��n� �=����G�='�n=�=�' < o"���b=t���튒���?=<�}��ߟ:�fo=:Hn=������"��������<�3����!��hI�Q����>��s7>��=	����罰r�Jz���L��<Hʻ<B��=��<6ֽ�߅��5C����=ύ= �l�VD=���<�}@<rW"� 绺p2�;��0��D����<<3> �!�����%��=�y�=�d�<t@�=D�M�Ƚ�0��IV��S=�b��A��a=��Q�`W�=���=��<tH=8h��v�=p�<�\Ž �-�9K�=�-<����� �,ҩ��R<��=�>�=0X��������_�
נ=��C���">�Q�=`��;0�j<cϡ=��;��>4R�<0%b��9�X	t���<\��<�Ό=ʵ=�z�;�MѼ����;��q��넫�8�.�ϙ��'�h$"� 2;�硼��=��3��d$=����p���h%Y���׼��Խ�7;+�����):<.m=���\9μd�(��P��ɷ����������!�<7�����@����=����H�h{$<r�B��(��8���t+�h�ü�� ˻8F
�V�*=�F=�N��^(��/��F[<ʜ�<=<���`�h����=��<-_��k��Pu����ν@����L�};�n�>�I��=t�{��5>��ު��� ��;�W�<t��{6�y'�� �:t�׼�H���[�=@ʺXq�J��~�=�f�$�� �[���D����=`彻 ,X;����D��< T���м�����\���ȁ�`~����&�JB*=*�ӽ�҂=�P'�]V���f�=ā=���ǽt"��Q=�=��.:�d���������u�s���T�˼��=B&=P�һ��=@%�� �� ~����ڽ�>Žp�<��H�R㜽,����D�w�޽x���'�<DB��A����= S����wt�=d1��d��=h�a=�N;=������g�8�|�R�P��hH=\i�<��E��p�=��Ƽ@�=�4k=@��:�}������!�<���ne�H`�<н����[�E=��=�]���Y�x8ȼ���;�&����K��so�O齽�c�(��=jƜ=����ؽ�
�<��
��R7��E<ԣ�<Q�X= ��:$_�VLC�0����}�=�/X=�6�<P_�;�$ݻ�����J�=���Q;@  ��^Լp4���>����\�W�x��=��<<�<��j=�!(�;E�� 4��ƛ��"=����(u�<�0�;��X� 9�<0>���< ��7�WV���4=6�;��ὸ-�k��=�D� ۵� �T�H���T�7�>��=��H�?N����d��y=���tq>��=,����m;Ӱ<=8�����>0[�;ټP�P��Y��ͮ8=2�=�=�2x=�9ڼ@�z�@�ݽ�������7��j̽��7#���.��������;設��!�����h%�<�����Q���2�,u�<���M��A����.��!"������S��ă5��3��Ô�d=¼�zb����;����lF���=Л��,��(�)<��S�(o��b�2�ߵ�t�%�v�M��J�<�;�P%=�<����o
�������<�1 ����<������<L��=`�<>���� �t��� P���G��/���k�Jv�=��q�X�Z�xRJ���� �Y���
�`0<��?�ý���<��C������=`<���fh��䳽��׻V�+�?m���6<�3�<�{Y=���`Ц��J����< c�<LN��h��<�o���j���м����81����8�?=�jD��z��(b�<s,q=����鍽�.����0=0ȼ�y�����@<н|a׼�D{3����=�U�����< '<؈�8�ż̛����ɽc��pD�;&a��o��.蚽Л�<OAý�1�`�<P�׼��R�\V�=d�ڼ�K���=L������<�'�<�1=pQ�� A�;�qּ'��;�q=��;�)�V(^=�r���8S���#=����6� �ػ��a��b��
'� ,p9 �<:�J��,*=(�`= f��8`;� ���û���4n��b�����,9�i$�=�%}=�m��'��|=ȱY�t��8�n< 7:���<�ؼ�y�� ��)��E�=0�<`��;8t����߼ ��>��]���4�	�K�V���h����=�"��u��E=�$"<�B<�<gm������bM��ɽ��<��P;�<@� �������#����=(�<^�p�ͻ&=��Z�"	ҽ�1���u=^c	�����P?����x�����=#I�=@W��q��HYt�BB=LJ�����=B�=$D� ���Y�9 ���>()���m
� ̽���@C��������t�Ӽ@v>8,����J= ��=j%���;�a>=T�(b���8<��
��H!=���<e==�D��`u̼h��<�ٳ��aѼ@1.�~�C��Cv<��I=~) �=�B=�36=�ڮ���0=n��)d�,kd=��<���zp)���W����������l�<p$<Õ�����=D��=4�4��ʻlM�`�B���:�=O��=xh� >�� ~����=n`��`Y;��o������L=~
�$����K�Q����=¹~��K=.�=�s���R�<��%=�Ns;�.���w��T�5�H�y�g�І�=�������<Ȟ�<�u=�����D�@!.���=�|���hN�"��4�ؼ �ʼ��o�o񪽀���T�=t���{T�P:<�\���2��@�0��[.<`�7=Iʅ=8�6�=�Ϯ��+:�q��H��<xȼ�1� u�l=ba�=߈���x�=RdF��]=/�r=l�N�'�}=�6U;������<�c�<߁.=P�a�D�/l1=:����<T���D��<�9Խ,�� 1���ϼ$�	����<bY�h�μoo=u�Ͻ�G= ��;:�[=�!���üH�+��'����;�޼B
M��8�= Uм�\k=��<�CP=@��T��lO�<�=�Zc�sE=�������<�TC=�v�<�-=@ <�L�������<��� ���8"&��f��b�=JS=�d�s�ĽPfN�8�ۼz�0��Z�<�{��Mu=ȪA<J�/��)"���E�ER�=�+=
O�v��=��<�ɕ<7!Խ��N;zS����t��=Ԋ�<V��=d�����:�8.{=���=({.���<@T������=�s��r�=(+�<��v�X7�<Nc����i=�q��!:do��`��;P��;�e��C���A;� �=��=�p�l�<4R-��&#�P�PCi=؇ۼ�;����Ӽ=,=�, �qt�=�#�=�4%=��<(�:<L\��C>���< �{9N弽����pYY<��%�=`D����<>򮯽�%t���O=*(Z�i3=*��=����C�E��=.:U��)��P��; �m:vz罐u����#=�����h��
���}~ؽ�=�4=�e�|h3>�D���Ͻn=�o�`t�m�=X�1� ;�:�.ͽ5����R��R��`
�\q�<���<������=�'�= ��;��<�p��|��B����
>�̰= �,��h+��<���;�<@��Оڼ�Է�x�<��=l� �F���� 4����;����h�~<<�=�!��
�<@�1��:�9���.gp�^�=w���`��.C�=\�K��,�=�p�<��!>mE��|���t��J�=BsH���v��]�o掽��M=��6���� 7K;�=HcE�D2
�4M������p���༖�=_��=��;�<s��=��a>C=�怽��=\�@�E�9�5=>��=���=y�Ͻit�=�.��"n�= [=�}z��A>�0���{��qh<���<���<�⸼�`��.=��N���";@�\G�<��r���R�\����\�����?�<�z���ƭ�e{+=c妽t�
=�$�;'=�6�1q��H�(�� �;�v�@�¼4�.=Зݻ��=%�<ԯ=�#H<��w���2=���<� ��EC=�II��A�;�F=�s<j�	=�I�:�O���,<`��;�Q��8����<b�,���Nh�=�Q�=XM�Z�t�PeĻ���"�
���<��+���= �;��o������=^^	=\?��K�d=��< r�<����� ��l�� >��K�R=|��<j;�=�\��xVü@2�<T��=�0Ǻ��~=H�[��H��H=\"����:=�͒<<�ۼ�4�<�}C�3<=(�=x�S<(�b<�y<p�<@���`���p� ��a�=:=q#�h]K<�y޼櫼0�<��y=���~E_�0�^�B�f=p�m��k�=8��=���<��<�=<�$�=H��<��<�T���7����=諒-a>�B��4�7>��ĤW��z�;�y��j�=X��<J���!��GV�=�7��<{�x~
���������9�<�/=(^C��XL<73�f�@�(�=�W�<��;؍>����Ž؆�<�I����G��F�=2i��w{=���|�PE��t�Ҽ�
�� vT;`�<(�<�z�=Ao�=�f=�M;��H���޻�^H�9�+>��<(j2�X���P������;ڌ=�=���|���>�sp=�2�V�����~�r��q�V��د���C�=ܘ!� $o��8I���D���˽J�/�r�>�N�IC������NԷ����=�ī��Y>ya��`�&�L���?�!>�L˽W���fNq�'�ν��>p�Ի&���<���<Nx���]G�B(���25���A���={`&>���ԉb=ͱ`=V�H����=L9��E�=��R��&=/��=N�>5m_=r���W�<	��e��=`;(<�9����>����t�Hl���<~�=�.Y��C ����< �˻�Ơ�X{�h�<t:���;.<)���;@���8~=c<6�}Iv=����Z/-=���<� =m�����;%� ����͍;x�'<����u"=��;(�<sJ5=Ь�<�C�<p�����=xx7<8ȯ��g=� Ἰ�%<&z�=�<� �<(�q�8<�&�<�S �`ʌ; ��k��j���
�=§=S����>��݂�lG4�H)j�EHD==e<h�<P�D�����T<� J��$�=�M�< �`<�OH=Gb;=D��</Q���ᵻ��A<��p��<���<*S�=��|���'���K<3��=@��:��=�z��~e��?�<ް�7�m=@��:�V�:@=�żw�7=�=�=�=#=�%2<|k�<@<<p���ر����=�F�<���Xv��1��`P��Ιv=P��=�{���0�x�b�*�=��;���=��Z=���<�0��
i=�*=F3�= �I<T5�<���N���/=`
��->h-)����=�������d�׼UL���3[=��T(�QS���p>�����:��h,�[;Ͻ��׽�K}=�ȴ;�Ix�ȂA<�d��R��$�=09�LK�<�<t>�B�`ϕ�	���=H��@�pY�<ďM��N�= H�{
�ȯ~<��!��_ݽ�O����f�<���=n�~=xt�<�����(�  �`%�;�	>�0���K�
���+J�TaB�f��=2����1Ž�A4>`��<oͽ��r�7����s^��,���|��H#i��
=-�<T1F�4�����5���쳼5">��-��ț�AH�����=3��4>�τ�����	��>����2��޷	��/���(>�Γ<�N�:�=��&;��{� �"��%ֽ93���ẽ�N���M=�&>�h7�I��=�
b<h�5���>�~[��E�;��]���<�u�="e�=�!<����ڼ�㛽�7;< K����y<�qg><������������=@��<��B����Y�;Ǽzz��_h���2�FZ��K$;�`���*�����X= ��㐾��=b/a�ûy=_�9=Hw�<8fp�xQ��O8��0+� �<8��<p��b= �;	=Ƅ�=��<��<4���͉=x���]�9=ؐ����;���=d��<  �:Ȥ�إR<Xa�<��&� ���P���n�/�&�N�&��=k�=��ý�⇽4��No������.=d�<�x>=��ż�5������`*<�d�=h��<t4�<�G1=��=���<]Dٽ�F3;d��<��޼@>ں��t<���=0��,���,�<�[�=���;^��=p�ļ�c��Y��WU��`V=h�A��؝<� =4(�n�T=���=_L=d��<�V��(��<HEn<BC������7F�=�3�<�)�<a#��Lt�X�8��l�=�Qw=H߼j�b�,���ۋ=�{S<D�="��=�	�p_�����=&x=.�=�uH;X�<�����f�'�<x�=<���=�!�<7_0=�3���K��ż�낽�lA<CZ������l�z�=�D�"] �dd=�sB��Vg��xe�=�T�@�,� `���ue�F!��<ޖd��L9=��=D����������.����� ׻�&�a9=:���
���-=�Z������Լv�� &t9��]<�OL<4̍�&�o�
x���u_���<}�=�pW� �X��`��6��f2�l;�= t���a��<�>������l�$j�F{��K�6��B� �D�>��巼��=^����T���m������g�:J4�=�e��$���5��d�=���̼�3��/��=8��<�����ҽ��o=N�a�����H�[�2N�V�
>\"�<`lU<�|�<`�R<hs�����$Qż�BȽ����.������=�{"�+S�=:o ��޽��=�ǰ<��ƽao�$ּp��=�m�<T@；���8�|���E�O%��p񱻢��=ӓ�=`�T��낼h���5�;hC�0�[��Fg��PD���¼ dýlכ�Ԋ�<�@����@�<pf��Ă���|�=0�9��T��5�@=xz�^�%=8�j=|�< ���l��<�ݠ�'ۗ�?=�,�<���6�i=� �: �?;�΂=`�� » �R;�Z�<�B!�hn����c<�C=<|;��_�e=f-<=@�X� 6����;phe<bp�|��0�'�Z�q�4���
7=C��=+½&�K�@p�;X`ܼ �};���<���<(q<��;�}س��X� ꚺ�D�=��<8�R=`�<� ���@@;9a½���0E�<p�ܼ�����弉U�=����(����<L��<8�n<
T[=:��=閽$Cj��맽S�=��a;٪{= r�;�:	��D<J�>�L/=@��� -���=f*<d�����μ�8�= ����~�4)i� ��|�üx�>nYx=�Z���SC����<=�6<<�=ƭ[=�ͼp��/=�S�;�f�=�}���4�;d+��0an�6�"=�R�<%x=�-=1�<�g�,U%����ʎ�0Ή�5����2C�F�i���==��ƻ0�6�t�V�}j��4R�K�o=|���/��zȼȐX�����@@,;]z��df0=Ui=���@Nܼ(;����z����^z��V�i�=qa���{�A=�*��B��LoҼ�8_� �=�`0� (���#�a7���b���ټ�6O<�q=�Ck�(�C�،'��ch���%��;=�x��T��/�=0�6��c� ρ�vt@�!���L�����H� �6##���=ơ���?��@���������P�<� FV���X��v�+���ƽxq�=�Tj<@��r��`��<Jr��ϙ���Ѻ�iǼ�L�=ج�<`�v;X�[�$;�<p�;�'i����;7���ټX�%��.l��T=л��'c=�@H��謽�Ǖ=�/=����c۽p�$4A=�<��/6����:����zO���Ā��DJ�=h��<H�}<�S� �7�
>�0����;�������ԙ��<���`���	�)=X����2��'?=��B�0���V=*��O5�� �>� t<�'Y;���<0ݻ;𽅻q�C=\���νlV= ��9����%= ���� �;�=\��@%��_�<H_��?�;�	�:��`4=�� �}0=�
= ���ܠ�<�!��K��P�ϻx�����=���k�(� <�<<�RR=����z5�&T =���;�{׻X$3<���;Ȑ��@���z��8A���Ǽ��0=*:�y�=�B���Ǽ@����7���W��@ö�@��(L��fȇ�-��=:�&�ؚ<|�<�V뻰��;xw�<�|>��r�����=�Ľ�͓< ��;W�=t_���M��G򼫠�=�S�<\�#�.�; = d�9���ͼ���=t�ȼ@h,��Q��O<��ӼKy�=�Iq=�/��^�?����=�<���X�#= T=�X���ؼ����l����&�=�������n�̯�� �c�����yN<�����=���8��<���<n�� O=����<���(��=��y��웼 ̓9:~&=� ��~�a��	=�� ���LR=��/���N=D��<R�`���5<�ƕ<�`-� %�9�f�L㼘�U<�Ć<@�r<�g����=���L>(���3<�~<ȯp<g+��m=�p�=h<�]*��都T1�<geֽv��=�;�=�\��K�֥���}�= �b� I;�҉<p�v���r<?�̽��.���z��'���T=]#��14L=�=4�>�|.	=pWK<��=0���������5��\���O�=܃ ���!���X:ѹr=/H޽�,�`-6;l��<�Z'� rI9Tݼ�\��nom���`��� �z���F=>�x��� Y�;6�+���Խ�P2�NF=��=���<DC��� <�����/�<���=d󎼀*<����y=�=n`�"{�=eI��z�Z=)�=�ys��gQ=���Hz'<3:'=O�5=:�[=h�;�����_b=Ĉ����X<l"ƼǠ2=�� �ռ�]��<M���q���c>=�@K� �� �<CY���]C=��J<(vW=r������ �i�(fS���;pZ�~��rώ=hDq�@�v=���<�g=��e:�3��=bY*=�!�`�?=އ��<��<�e=��X;�]4= `�����:�K,�g�G=�)����f�@�M;�I{�*y�=�H=VZ�[�ŽPQּ���+Ҽ�[	=$벼�n=����rO�|�ͼ,��,��=Z3=�1"�'��=0Ǥ<�*�<}�����<Lϥ�,���9�=��<K��=x�'�\�I�?	Z=�|�=�D0�8/\<�-t<�i��r�=hfx�XG=6=�kK��=(i����f=��o����<�|U��<P��;0]a��#'�H}<�Ŗ=�$=`��;<��<���>|=���&��k=�N|�{х�������<����$�{=`��=G�:=��W<Pg�;)���S>^�<���;@GV����$<�u����=�L�[|$>\5��28�� �T9Bo����<>�8=��G�>�|�_��=O|��;x�����@���@$��vA�ȦS= �g��&μ4���Ƚ��=8��<ҷ&�AX>�[j�p�{�_�<ƦX���۽̚�<8���as=��Ͻ�н��� E�@B#� d<4=�< �����=��=�i:=���e�&���<���0;>{}\=�T.�`s��$x���}�:��y;4�F*:�A�<x|.<U#����6�BZ�l[:�0ǆ�������<ҭ�=�e何b =��ȩ�<V,X�/鍽l^�<O��G���p=�z��r�9=�V��r>cz�p�~^���Q�=�R`�~R�>_C��Q��x�z<��	��#��Y��=,v�Ш���:��=r�����7<�N��=(��=�J�T ����=.���܌=��A��w�=4#���ni=�n��8��=;�=��½�[�=�����a�=��=�����s<>��� ����h=O=���<�Sټ0�e�h��<H�����;:�U(�w=*%t�h'C�\Ս� jһ�����{=X�	�`�;<�<_O���= ��<
�=P�ʼ=ؼ�,�dyƼ�����J<ȣu�ga_=�V<;�w*=�r=��)=8�y<Z �^D@=lN�<�t=�<C�Ђ�<	�k=�ǈ:��=�A
��"�<���<���<�R>;L� � ��<�¼�3�=��~=j�f�����Q���N��^y���
=�|��-v<=@ܼ�,��^��@L8����=�v=8��=���<L��<_�ߜ<̕-� ��:J8=Ӡ=�D�=�r!�T{#��ې<ַ�= Ҽ�;=02�;���c�=lG��H�(=�F�<d葼r�#=��(��$K=���<h��<�c<@��;��<0鮻���7�;���=i�4= �E9��<�����1�H�#<L=4O��ʌ"����Ks=��K��l�=���=�`�<@�;��=��<]w�=�Ȣ<t��<@T漪����M�<hgʽ.�>��->(��<qm���6������!?=df�<��K�a���Zd>ހi�����K��m�����ǽ�Fo<	�D=t-���;�)E���&���>\�<@X����s>�!	��d����9;U���$�+�jT=�w����=6��j�	�*	�H#��0^�����D�<C�=.��=��= g=�,μ[�c�(�<~h�t�3>���� ��';�ں�]�轚�=��tS���>��;#�X?���H��Qg�k���;��T.�����=����@<�F���ʷ����2vt�q>na��Eg� '�9�3����=l����E>X�0[ݼ���>Y�/
����d����5�=���;@)2�ϧ=�;�ٜ��[�;P��O뽺[��x��+l�=�<&>^�&���j;T�=P�>�h�> -A�9^�=^i1���=��+<�\)>X ?=T��h�<Ѝ��S�=@ku�P���^��>�Eཬ,ü�K�:�=P
=`��h�z���;�=�|���h� ���<LO���7'<|���@�{<�軆	= �?<pk3�=��e���<w�<�o<��^�hSV�XA!��#�0&����< 7;�=<��<���<e�+=�<�a�<�ۊ�$\�=�T�<� t�m4+=蘺���<��=�Kp�$S�<htj�X�=� =l��<8��<��� U�:��Ǽ��]=��=g:����%��`�P�Q��慻�:=4G�<h��<���t��z�(����<	)�= G9<$�<Agk=?7'=C�<BA���s<��ǻ@(ݻ�ޕ<��<��T=�NU�,/�Hς�Y$�=`D�K�w=pt=�@f���<=��QG=`!�;@��;1�-=�C���*=Q݇=��U=���< �D<,�<�L�;X�ȼ�7i:b�=�
=���;`�0< _��������0=�w=x�N����:��E�U=\#�<�W=�Td= d)<����LA=ښ.=��f=�' <��<�*��T�T�<p�X�h|>6Z�����=d�ʼ
d��띇��V����=p�3��l��r�<�P� >���N�`���������R=�s�;|�d�8H|<m�.�§4���=,o�<�2�<�5X>���\�"�̈��K��&����<0U���=V@�Z^� 􀹠ѼH���n-��إ;FA=�Ss=��{=Zf=�n�I_=�2=��!<���=s"��1��=�T�����$�;_�=dxQ�N0��+#>P~5�����)�f #��iI��/��J _���Bs$=��R��@'�2����j�������;���9>QZ:��A���x�x���=��f���>��"<�裻s 
�B��=�ֽv<|�bf���ݽ�>
N=h�<M$u=l:��׎��ں;p�ٽ�ܽ1e��n�����=��*> �T�N�	=��N;R��>xd�P`x=�4�&�=�/�<���= r�92���r���Gֽ�0'=�S,�Dݽ��$j>�Q��\ȼ@����	=�L�<����:�4ݪ�X��jW��R1�03<K�����;D��h�<HC[��
0=��;V�W��L-=D�8�=�y(=pd�;d��#���a3�:/f��p��Z�< 6-��]=@~y<���;�f=x��<X�<�J���<�=�l6:p�5��&=��� �l<i��=�K;pY�;���n=0?�< �.��[T< Ha��X�f����P=�Y�=�Ӿ�f[F���X�n?�� ��?%3=�A�<��<�2��v�Hb��(�<޲= 8�;JK=k�;=*$=���<��t� �Y<���;<+Ѽ;���}1<��h=�# �|�����ʼG��=@w;��M~=�����:I�`CQ;��*�F�"=�޻PQ�<��=�H��x-=�x�=�̀=xr�<�$<���< \<\Z:� jm��G�=�<h�L�`�q��K.�^���1�=�c^=�8��J������Вg=d��<��l=��f=p�V��=8�|ln=�(=�mb= 'c�dx�<@tq;\�X����; C�:��=L����%=�E����b���`������A<�U�� ��@����=��"�՝����r�r!���/��$�=��#�(|)�P`<�Ƚ��&�R=��ѻ<=C^�=*�۽`Tۻ�̒�2�L��8�8�h�0����=�󤽾6����<(�~����A�� �ȼĺ�<��<�-<����<;�Nǽ��<��<.�e=H�U���L���콰	��
�T��=���D�]�=�!���5=�d������  �-L�̨ȼ\	��0����=�*��2����Tٻ]bŽ���R�=rc��"�eν�߮�d�������K=�S,=@\X;~W�1DU=o;��r��:<����(|>��D=p�=�P=�k�{�8����̼u���XH���!��`]Z<}8�=r�4�F==��}����8�=���N���齘O�<L�<���<F���_��P�K�������ؼp�9��V+=y��=���E���=-<��W�9м���R{���s��ެ�P�|�hn�<�?޻����0f�;�D���^��~F=��������q<,Є�tĻ<Bmc=���� �1: �k;pvl�yݟ� {6;8��<�$L�T6=Є&<�n6�;T= >� ^Q9@`���=@�ܺ��>;���;�F�;P�	�i�N=���<���;�}C����<$�<�8� h�;8�����a���q�̘�<�DZ=�ɽ�8������� ���lS�<h�k<�w�;�Ќ��Q�z�	��[-<،�=��;�3I= ��; �P���;�]� ����:�<BܼH3}��4���q=X= �HO4��徼��O=<��E=B޼�0�@��{��4��< 
�;B�u=��\<t��xJ<7��=�N=@���Ծ;���<�!<;��m��ﺼ��=@>�;���Xe��e׼��(�@$�=�M=4:��Ή� ����S=@ك<�9=;98=f�����J=q|<��t=�j�H*d< �;�>S�`�J<X��<�lN= 9���^�<����:�.;V��ʝ��;�#��h�)� �D�=^�=@ƺn\
��:��+��hJ�Ŭ�=�k����� [+;��j���ܽ��=��ǼޡN=˃=�������ј�v�'�c3佸#T�P�׻5zs=������@�+;���&����\��^��@��:xX��E�;�����p���r������<�q�<$h���<�a���|�T�~O:=ؙ^�j���O��=!ؼ��ͼ��&�����ɽ��� |��8��DS��n�S=b���>|��P��;��˽����d�d=�i�`�<9Ľ �2�~)(�í�DB�<n�= Y�Z��\(�<�~!��8�����,w�p��=�#=`��<�y<�"�:�<���m��A0;�^� �������#��=*���p<�<V�:��M~�8��=�@[;�;��>֙��E�:��*<����:����$t��?��V�/����<u�=�,= ົ�^��Q!�@�R��m�� iM��tD��./�pB��j򭽔Q\�C=�< z��n�<pA�ح�<L�<05�;�x��^��S�<8���=�됼���;�]%=p�߼�Iǽ�Ϸ< N79��);0$< �;�H9����<��!��v��X=��`�;,=�I���E=D:��<P�<�7�; (�<H�W<`�%�X�u<xI<��Hֻ`T�<�����<���p �(�<t�<�U���-< Q{;&�������I#�6�;SZ�xa�<�Lؼ��+=��P-��,�:�����&7�@�G<�a�*��fi���<N���)�;D��� ��9$���`�O;n�!��ּ�k��g�� �8H<D*�=܆��h�����"�=�ζ<H,��7<�U�<@2��p��xJü=�n=�W�0!|�������<����K�=7g =���|[ż�����~�;8%<��� }�:·�*��P�#���wC)=䄄�Xw<*
dtype0*'
_output_shapes
:�
�
siamese_3/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*&
_output_shapes
:{{`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_3/scala1/AddAddsiamese_3/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_3/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala1/moments/meanMeansiamese_3/scala1/Add/siamese_3/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese_3/scala1/moments/StopGradientStopGradientsiamese_3/scala1/moments/mean*&
_output_shapes
:`*
T0
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
"siamese_3/scala1/moments/Squeeze_1Squeeze!siamese_3/scala1/moments/variance*
T0*
_output_shapes
:`*
squeeze_dims
 
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
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_3/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_3/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Tsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
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
Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese_3/scala1/cond/Switch_1Switch siamese_3/scala1/moments/Squeezesiamese_3/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_2Switch"siamese_3/scala1/moments/Squeeze_1siamese_3/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
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
 siamese_3/scala1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
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
siamese_3/scala1/batchnorm/mulMul siamese_3/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
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
siamese_3/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_3/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
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
siamese_3/scala1/poll/MaxPoolMaxPoolsiamese_3/scala1/Relu*&
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
siamese_3/scala2/ConstConst*
_output_shapes
: *
value	B :*
dtype0
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
siamese_3/scala2/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
d
"siamese_3/scala2/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_3/scala2/split_1Split"siamese_3/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_3/scala2/Conv2DConv2Dsiamese_3/scala2/splitsiamese_3/scala2/split_1*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*
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
siamese_3/scala2/moments/meanMeansiamese_3/scala2/Add/siamese_3/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
!siamese_3/scala2/moments/varianceMean*siamese_3/scala2/moments/SquaredDifference3siamese_3/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_3/scala2/moments/SqueezeSqueezesiamese_3/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala2/moments/Squeeze_1Squeeze!siamese_3/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_3/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_3/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    
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
usiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
Nsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
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
Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
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
"siamese_3/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
siamese_3/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_3/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
siamese_3/scala2/batchnorm/mulMul siamese_3/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
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
 siamese_3/scala2/batchnorm/add_1Add siamese_3/scala2/batchnorm/mul_1siamese_3/scala2/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese_3/scala2/ReluRelu siamese_3/scala2/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese_3/scala2/poll/MaxPoolMaxPoolsiamese_3/scala2/Relu*'
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
siamese_3/scala3/Conv2DConv2Dsiamese_3/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_3/scala3/AddAddsiamese_3/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_3/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala3/moments/meanMeansiamese_3/scala3/Add/siamese_3/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
3siamese_3/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala3/moments/varianceMean*siamese_3/scala3/moments/SquaredDifference3siamese_3/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_3/scala3/moments/SqueezeSqueezesiamese_3/scala3/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_3/scala3/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_3/scala3/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese_3/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_3/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_3/scala3/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_3/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
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
N*
_output_shapes
	:�: *
T0
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
siamese_3/scala3/batchnorm/mulMul siamese_3/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
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
siamese_3/scala4/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
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
siamese_3/scala4/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
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
/siamese_3/scala4/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_3/scala4/moments/meanMeansiamese_3/scala4/Add/siamese_3/scala4/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
"siamese_3/scala4/moments/Squeeze_1Squeeze!siamese_3/scala4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_3/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_3/scala4/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
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
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_3/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
siamese_3/scala4/cond/switch_fIdentitysiamese_3/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_3/scala4/cond/Switch_1Switch siamese_3/scala4/moments/Squeezesiamese_3/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_3/scala4/moments/Squeeze
�
siamese_3/scala4/cond/Switch_2Switch"siamese_3/scala4/moments/Squeeze_1siamese_3/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
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
siamese_3/scala4/cond/Merge_1Mergesiamese_3/scala4/cond/Switch_4 siamese_3/scala4/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
e
 siamese_3/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/batchnorm/addAddsiamese_3/scala4/cond/Merge_1 siamese_3/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_3/scala4/batchnorm/RsqrtRsqrtsiamese_3/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala4/batchnorm/mulMul siamese_3/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_3/scala4/ReluRelu siamese_3/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
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
siamese_3/scala5/Conv2DConv2Dsiamese_3/scala5/splitsiamese_3/scala5/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
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
siamese_3/scala5/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
T0*
strides
*
data_formatNHWC*
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
score_1/Conv2D_2Conv2Dscore_1/split:2Const*
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
U
score_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*

Tidx0*
T0*
N*&
_output_shapes
:
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
value��B���"��T���`����<,eʽ̚&�(���_�=ą3�0k�=HDZ=���?e����l���ٳ����;�ā�P*	<M@�=��/=�$���`;��,=��J�T����?<<$߼�3'=�ߒ��,O�Da��|]k=�T���X���<��<l�ͼ¶�=��V=^4�=�T�HT<Z]���M&<$0�=�?�<hg�� =���<82��T�� �N;�Sd=�˽�xH�=*QN=�?o��H�,��O'�=7���Z�<��=�0=X�<e�
ȍ�+r<J�J�P��;!ҹ�A�<��=��ȼ4 =v:R=�X.�����vJ=ЛS�8W��"׽��=l��<R���i�Оv=�^̽rւ���"�@g^<h�p�`>�<H��=\:��d����l�.%5����$r��K����r�=�~=r7�xϼ]͈=8���P�=������=�&�<m��P��=�&���=@��;��a;r>2�vw9=��=�b�Ԣ�=ҏ�B >���>(y�<�M=P����̻4U0�0�,<�ue=������h�n<���M�&=�5�VY�@v�;xkM<x�q��r�<�]��b��4
ļ-+� �<�$Ļ �H<�y��Օ����r=��F��%:=��»���0<t�0�\��~����0��*�;�:���(<�u�; Pd�d����Z�<�1=@�B<X�A<�Xf�`����m������5���::П(<��ڻ �;���<�𼸪Q= ����ŻLM��D����,��pR�< �:���؏`<��_<����"K�X-㼌�I���８��<}�'=@��� �ڻp[L<h5<d��<�X���9ͻP�<���;���<��м�5	��BԻvG/��5�������)�(�_��	C�B+-� ��; �мȢ���$a�ZՖ���9���p���GD=�"�<�ර������Ӽ(�I<�=����m�=���L�˼pu�;²4�������� ��;�^r���*���3�pd�D��8�D��<���<��y� ��Ks<qz=������7�:���hS�=`�߻��=^(�=y^<�Z��8���3��|���`�;
v�X�<�X=l��<JK����t��+�=��!��r�� 4��. =���<��:<�G�@�<�-=@���(��x��@VK��7��>~=l�=�Ų=�i|;���<4d��(!�<g�}=��=�I�:�.=6yx=A�x=�=��=�3�=����ڳ=	(r=���< �8�^�;\T�= P� 'y:�d\=���<�_|=׻νL��<�<$ۻ��S<��m� �&=C�< 5��(Td=�b�<`�<2�w��E=��*�����`��j$=xM�< l�;�	�<,�=����7;У�����<Ys�� �9=�=����	t��{[�䷸� �9��:�3��8����a=��������m=(�P<��B��O�;N��=�N=8�¼]�=����6=��==��<(׼ϔ�=8��<P��<KD�=��`'c;���=@�;�ê=إ�p8�<�ш�����P~ƻ�n��L3���:�<���5�>=�P���+���(�����L5����X;�	�A,� �=�;�ܼx�R���3༤a�< k�8�<P ܻ@��<���<5�`�z��·��,��gD<�kC�`/��8v5���<���E��b]�`(����;\Ƽ(������; @��pl��'Z�w���S�ȗZ���<@�L��"���0U=X�ɼ�c5=����Z=(U�<fI���ɻ0V�;(�&�:< 
\;'�#=�^�pA� �`�D�`�NǼpC����:=�P_�l�д,<@ٳ�9�<W��]�<|'�<��	<�L�;V���׾� ��j�9���.��>��3Ǽ �k����:�	ټH)���qȻ@��� �.7�ぽ������� 6�9��<`��<�ؼPd'���p�;�j=@*t<H��=����Ӣ;p؁;�uż�I��Rü ���X�ݼ��  �� ��90ñ��,�<��<���IW=dI�<��F= f���
�;H��H���6=}�A=��=H��<�؟����=\���T9��T����m�<�%}<(0�$�K���S=8-
�!�$=p$���=�~���=8����m	=8p�<8f=�n{��>�0ŀ��� �೭<���=�F�=�Æ=(!=f�D��<��)<�c= n�;���<���<��
>Q��=C=jR�=r	=�=Q'#=h")< Q<0�=�su=�x]��o��F<Q�<RI�=K�Ͻ"����:A< <|<�ŏ<�2}���==�¿� �E�= ��N=�哼�$=P��<��<��K�Hi�� 뼈�<�"=��.=� �Cs^=ԁz� b?�8���I�=��&;�>�<@��;�h=H��<��l= ��<4�d����;�g1=l��� ������<��=`7'���;ʟ�=��`����<��=`Y<H;�<�r�=�:<� ���e=b��A�6=��t=��h��<�����;����=�x%��L=�y�<\~������`&��R����=е���3=pc���ۼ����Ϫ��)�X"q�d| ����:@��=��<�%�^����|� �:~�o=��ڻ@��������<���:t���y9��c�������=$�C�lS�@ri�l��<��*�V�'�Ѓ�����h�缚�"���Y�<�Wh��X�;���@�z�nP$�����Ђ�;��D���{��4�=��<:V@=��d�fz=�=V3(���<@���P^�;4��<��˻��A=�����< �f�����vĻ�t��V�K=��м��˼ ���}��\��<�m��s�.=@�;@Ԩ���)<�u*�`�����;�ۼ�8K�D��<𲢼�RF�5<X�b��0�@��;T⥼|9=���@�:x�Q�|��<�����.�<`�ʼ e�:��� ��:�Bt=�:�<(O=����<0�	<��L��+|<PA��.����<@_<��<p��;`k;�x�%�=��=,���$�f=����h<�V�8c0���ٽ��ǽ��o= *r;|P�=ৱ;(<�Y����==�=�][���< �<����I��(6\<��F���޺�<4�^ �=(�g�oܨ=꺓��=@n亀�úȣ�<���=�N��H�N<0��;8�T�p&H���4=={2=���=���< �<;t|�<A޼��<HN<@~�ɶ�g��=M��=)�=H�#=��=𹷼��C<��H�J�o�H=�6=���@c!����8C����X=9~��,lo�Б�;P]�<T��< D�:ʫ =��}�h'<�r&�`�����:<�{q;���<��
=��e=�ڟ�f�n��s��%<�R�<X����;춈=Z��v��6A�19]=����xh�<�=}jt=��G=/&�=s�:=�0�H�k���!=������<8W
<��2=(h�@���t��<����U=7�=@徺@b3<��=����\f���z:�3���{H=@���<��V;����e��r&<(E�Q�=�{�<L�����ּ�W�<����^�<�l.��L�< �~��3�Ԟ�d�Ǽ����xiL��&��=Z�=@��:��8�����0������;RS�=��E;P�O<�c���a껀�y< F�@�ջ�ч�@�(�t��<�W�����^�:�#=�é��d༈�6���D������;��c˼�X&=��:�l<pc��@&��Ӷ��dT����; �:��`�	�=��<��X=����tԸ<�O;=K��\=�ϻ`�y;���<����"=<����=��C���D��JI���k�b�Y=�t����еM�H2(�b�;��&�e=� ����:��#<��h<@�;��;����6L���=�j����;ؕK< ;������a<���}=��&; �d�PC�<4�/=@��xv�<0���� �;�@�h�U<��3= >=РL<0��G=\d�<�>��=@�<.j �Ք=���<\�0=�)�p��|��b'?=�Z�<�ݴ��Z�<�cļ�ؼ�O�(�i� �V�n��4�< �;��= r(����;�8��0��<���=˩���S=��<@�2<>hH��� =2l��J� �T�$��=�����=ܠ�5Y<8�;�_ێ��1X=F3n= �����6=�Ee=�C �����P	Ҽo�F=<��=������<�|�<�� ��������<`Ѽ���0+�;�=��=���q��=�H� ߀��Н�@,�ιC=\�(=nq�ܸ�����#_�|�<�� ��x��k[<j4=`=t�����л��}���;�����A<�"�0�Q<�\�<�(�G�< �7��B'�����	; &��HC˽�tj<|h�<]$�����N���P�<����`�<���;�9׺lX�<f0@=��s<< ���D�,�<����	=��=��=�U�@�m<4+��M��L�j=���*�� d�;Rx�= 3�_ڊ�O��&�y�� w=x�ۼe(5=Y��@�m� �T�\�L�x�	� �9�}l�Ȗ���N*�t��<�����"]< Q�����<@�6��� ;��j�,2ݼ<���d������=r70=d�ϼ��9�J��`
$��=��Ai'=�Qb<��=���r�� ��<04� ��:nԼ�d�p���L��0����1�Y=�nļ`<��pË��랻�?H���>���+� ��<`tx�0�j<(ݼ�+�� ���!p���; է;E�<�I=dT�<��i=�������;"&=Ec����< ;������P"�P�Y�h'�<T��&o<`��x�#�T�#�h4x�m�x=`���ʻ>��V��`�h<���:,gz��;=8{��h![<H� *R9 �:��軆�
���I�<�vL��;@��;�+!<"v��@��:\���0= �: ��tE=%+=̊����<�+�;�,����)����<���<��.=p��; -v;�
=��<l�ּd�<\S�<Y� K�`s�<x��<�L"�~��s7���b=،.��]��ѐ��
�#G��L0��࣡�hhݼ��ݽ�����5:���<l�ݼ��ռFN'�d���0�H=����h�= j�����<��ļTN�< �8�X���h0=�nF<�ֻ�xK=�0A�|:��*N#����8�3=\z
=$��.@==��=�r
���x��r���'=��L=v��Pq<_C<J���zP��j_<~�VV��
u����<��Q=�ML����<�򏽸�0_黀pU���=�]=�Ľ�Ƽ�4�ĝ���`��:�< ���,;}l=E<b�*�\r��x��p邻t=g��ݨ<<O�@-�:���<�k4�d�;�����ܼ`�������^�������8�輬ͼ'�Ƚ�V�;P�/<Fn��ܽ�^@��8��&�����>��P&D�\.���?�<��.����<5B=�;x V���y<�)����6���T=�E�����x{]�jQy= �z�k7��!5ŽvW4�Wde=�����<*@)�HY㼰�;n�Ľ $r:Ƶ^�X����5���6��,�<�G!�8�r�� T�9��ݼh�<��/<�]� ���{���^9���6<l��<V�L�CΜ��d�8����}���_�8<D/e=�w��d��<�����,<!��m�4�f����t���5���=^�_��o���ڝ;PP}���?<陏����� A)�x�>�8�<�K��-�0�<�¸��@��:'�"=<�輀[S;�7=�Շ��ڋ�=M7����<�2���d�8�_����p�=<��ʼ��@�`����1�vr���Z��]=P��;{C��d��`t=0}��%ٽ��3<*$��=.���P;��p��(���2{�
�8��qX���6���[��s���#�<��ν�s������|{�<Xpz�,N����<�q�<���x%�<������ļ탨��= �{:iqa=��4��<�A�<��z<r�O��< +;�4D�2��̣ͼP��.<x�\©��ᖽ�5=�,I�J2S��E��P�M< ���,ڽ����@�����=�,���>�= m=���}6��ri�(��F�;�~7������=��=hh�<��@o���W=�.���m<�\�:�W�;���<�5<Rvl�޳~���=�|���S=lڏ=��R<�8Z=��F����=d�M��D�<������<��=�����+�T8�<��׼����p��<�U=�&�<����_�3=�]=P�
��˦;���N�>Iڽ)�f=4��<�đ�D��<Cꬽ<����8=�WQ��̹<�0s��4���1<�W��I�=\q�=d]d�I����B=��f��8����Zđ=��|<�<�Ƥ+��X#��YŽ0A��@N�����;�Z���%=��=��ż��½�3��^�I���(���r�RMǽu.=���~n<�8�=���� <N;Px<d�=�׏;�	���r=�Z�x�a=(@�<�^�:�.a�0��<�*�<z�;�Vy=��<����1�=�dW<�:�;�Y��h��<htǼ,��<�r/=`D���1����<�\����=4似N������ ��; ��7p��84&��V��z��F�`Sg<tJ��,r�<케+6;����([<`5�s1=�^�h#O��Uջ@��:I���< _Լ�M< @Ը��;PՓ;�!���y��P��<hU"<@�!<�) ;��;@Pa;��лR�N�H�0�`���p�;�|�;�<�<���c=Ȑ��P�4=0Z��~Ի�*��T���I�@}R<E�<��;���<��@<Ă�����`�u&� 5�̏�<^�0=P�� 3<@�ź��<5= 2�<ے<���;�V���<�-u<��̼`�ڻ�����ȼP�����Y��RK; d���%�hԝ<`�M�H<주�y����HW��3/��~�<(Ck< 0�:��غ����͆;_�=0~S�#�[=pj��p �� 0�9b�G�H��� �z���b<�I������ j�;�D[;�k^;0v��6=�=�!����;xI�<X��<�벽���l�b��L�=�����=�~n=��i:��0�4� �K;j/<�"fR��ȣ�Y7=�d�=�?޻d����&��d�=�@w��!=�̫���$=H'�<��<�L>�@َ�O�=���X�g�l��<�
=���<��L=P�<�n�=Й��m�-=
C��䕻<h�i=�s ��@��6c=���;���<��=7l�=�X=r� �k�T=MN=Ht<
�=h�<�>6�����<�=��<�p�=����
�?�&=�ޣ<8p�<��F���2< ����Q<<�=G=��m=����yω���^=h�������wĽ�p= ,<��<�.u;��<����8���f��L�<���z�_=ڝh=�����4�8���,)����=��g<���j{��{=TӼX򊼪>�=tɀ<����`F=��|=��<h�<s?�=\=�UD=��t=�+�@na�P%=��J<�=�(�=@��< �g��l= ��:"J=�T�v�r= ��< ���8w��ao��%'���9=�p����<8��
�@�F3J��� ��J�*�L5�PxI��>= �������L/�lV�<�A_<Ɂ =���6�`tZ��=�G���iR�\�H_%�T�ټjR=@x�������Bǻ���;`��Z�/��3Ǽ��;��������:�#�L��<�m�:� ��;-� L2���6�`DC����;�a&�R�"�!��=�r˼��<��5���=�<�./��4� �S9)�=�|�<���;��=Է���<`��<�o<��<@�8���=v�C���ݺ�l�:����l�	=�_":zo=(N}<��ȼ�s<��u<�]<��
�;�h��QT���<P_���3;J:<�.���<�
M;�!<��.=Hmp� Q��ؑ�ȗf<�\<@��:�� "<`��;8X��C�j=��K�WQ=h�8�<�ݻjG���b:85���T�0��<��L< �<�9�<�n=�M�G=;�H= �p�Ň=��1;��;أ��`���Zڽ���@��<��1=��= ���3¼D��`�<�쀼�R������^M=��<�Nx���J���U�=`8鼗��=���I�y=�0;܏�<�h�LY�<��<=�!=V��hR�<%�X�Y<l��<B!=��=�[=5�=�dl�0Nd<l'�<`�0� �y����< A�9F�=4�=���=K1=u�+=���<ty=�	<	�<P="�= �p��5����;Ġ�<zE�=���P�3���=�}=@ԭ<��%����<Ƙ���#=��< ��`C�;i#����]=���<x>H<����n��p�鼂#=h<T<��<:,�@��:��W�ܟ<���"�=�y�;�G��/V=&�=�=<~A�=��4=��w��2��&=��?� �`� K�<� =���� 0�<ߨ�=D�<�d=�=�3$�hT1<�h�= �ܻ0|�;('�<��q�6=(JK=H\= �Q�P�w� q�:��b= aD��Q�=�&�=�0v���A�Nꏽ�0��O�n=T�ļ��<���:�hc�7#��T���x�kP���r�<T�<dV�=�0û`4;��~�v>=m�_=�=�n�㌎� �;q�=����`�<Rkq� tƻ������=Xzc�x��������W� ��;D��vf��k;$f���7�B�r��1=�BG�+�l����<���8V�`v�;PH���9���X>0w��<�}����2=(�U<�4� �u�������y=�M8=0o�����<xt����=>XY=��Z�!'=ؿӼ� =Ͻ��\�;@mѻ��.�Ц�<H�<0X�=���<�t���<�?=�+�;( x<���;I��@=p/߻(x<���<L��,�<p֨<��<��= ���=ȣA�Ac=�+g�8����(�(°<�^<�����p=<Ϫ����< W��)�W=��L��������<����C�����=�I==��<.A=��= (���Y=��=�}j��7^=�'׼����3�te��l[�P����3K= &�9(=�=8Q�^g�Q/��l�<�O<����i,<�+�< �;�5���ʼ�&��z�< 	�zֱ=,����̤=*-�@������ ���[<;��=�a߼"%=��w�@胻 �W;�c�<|�=���=���=��S�p �� c�;x�V��0s�PR3�d��T5�=��=���=�<o��=����%F<��P�8�� �=��=�Ȼ�˧��g��@G�;:��=�?����m��N�<��(=�=^'����<1ܗ�(�.=�sv�`��0�;e���M9N= �F<��7=�W�I����O:���=`�u��ܼ`i��0(><4]��h��h�)�:�H=��4� ���BȌ=�*=�=9�=��q=��x�C��$N=�j�� T ;�` <G	=�0� �:�K=�?��*�=jC=��뼌^�.~�=x弰�P���r�魙��V#=��D�̉=h��ȓ����<��<8�����=o�=���ȚS�>Ǧ������Re=&���;ֻ��b���a�m콬���56��}��◐��bu=TÔ=XX���}�<a$���$=e}�=H��=0ð��姽�W�;�<H�]��<=)�����;�N<��1�= �#��/׼`����ڼ��-;A؋��\Q� D�;�D���1�֩T�4�b=�{>��R�����`��<?���P�@K��A�喦�?�%>0�X< ]
��l��K�=���;�G�pF�;XD�Om�=�!i=��ۼ���;pY�祯="�=0jW�i�P=�ؼ�s�<3�ɽ0�)<l���gc� �g<�]�<�o�=���<�8��D �<
Ռ=tY�<�ڠ<�s�;���mb+=���x��<P�&< ���<h�<�ց<;�=t��=�E���!=\�0��EH=���(�"������<��#<�oM�I�&= �� '��`��;֓z=�%������<�C<ȷּ�r
>���=T4�<l�<qс=��}�=��_=�ʒ��h
=�0-� Ţ�� üh��,�i�+��(=�<�=�q���M�Ǐ��xv<�>�<4���v�< �`:�A<�{'��#��Ƙ �p�ɼ �;��P=ĸ#��~�=Rɒ�4��N7'���ý�l�<V�~=�5��2<]=���<5��B�Pr�d�a=Q�=hE=0WL� ��� }S��Z��`����.��F��P�1<�=�K�=`����ƴ=P�&�p6���컸e'�giz=�<�;\�������F<�k��y�<^�0��`_��'=nЊ=��=Z'K��<�����<8����'��w��l�E� =� ��"=�߼��|�0���t<��мX�g� �ι !лx�}�R�2���� �L<N-Y���s<@$�<p̄�x.�<,=N=���;�c���]=����l��<�9�<�c<ި��pԮ;�::<g+��<��=x�����/���<�y0�=t�s��,�����#젽�=X�r�
=�8N�Ƴ��	=�tT�`�ڻ��=`d0=�c缔�=����e��E�=�O�a���2���?�н4�=�`fe�E-�����j\�=lZS=�C?��Y<���5�<�x =f�=p��J*v����>���<�\�<q*��࣋��4�m��=���:tk�������K� a`�Bh���9����:�=���5.��^���U=�k��xr7�4���]X;66N��9����\9���ek����=���<�Dw: ��C�<���;�\��;�;�E�G>=�\=N�-��?���s����=@ =��)� n�<��鼀ͼ<�e������ ���� 1����ջV&�= f��^I����b<e<I=���< ��;8����ڗ���<���1|< =��㻐����;�(�<�J�=����U/<�j���$T=�!&�l&��f޼�h<�湼�)���<X�ż@�D�P*�<|�i=������Q�<�c�<fG�W�=*�=h�~< ����h�<�	r��y�=xE�<�Y��'�<TiC�b9�`���D��HV.�b���Ϛ<�F<�}�<ģ���S��(��`Cb�PF�;4��p�<X���aP<�߻p��;�:μ�1���=Pw��S����=����y���q�N�*�$��<��="��0=�^o=�?*��B ��z�D�<A��=pD(<��k�Ht����޷G� �&��D����������z=��=��S���i=������𯬻8"ӼfnQ=$k�������}?����<��C��p����9<�n�b�=�]�=��d<V�f���������x��Tj1�@v;��ּ�/�h�z<�G��Xu�<��4�j�7��"q<hܰ��?]����0dӻܳ!��x;�ᠧ��I:��Z���t�P��<.�H��@r� Q��ؐ��xI/�$w�<H!i��w$=�y ��w~<؟�<8$T��Ƚ ~�;­�A}����`=꥽P!W�b;z�n��=�����Q���榽!,��d��<6�����<����0��Ru=��߽`,[;t��<`�p��\0� �)���^���� u�;�^�$̊����ԏ��j}q��FE�8����et��hV�GN=��=M��ƹ ����ػ����R����< ��8� �����)R��Oj;H"k��D:�(мH4� �:@�U��zw���7����<Zp�vK�|V����I�$!�+���,f����<T���p����@����dڼp&��w��d¼�1��P�g=��H<�ZٺW����<h<��0J�;�������:�[����u�a�DG���=t�ּ��.�@
�d@*��ն<R"v��ۂ���3���<��"˼XFv�� :=`�μ�=�Y����&<H�c�������j�����$�z�F�����દ� ;L�g�Ժ���V��[�=��(:�����/���=�p9�`����Ҽ�� ��π�x��@yĺ��;L0���<�O=��;Z�u�P<*<lJ�<��=�� =Hi=������-�DG�\P%��u=��|�W]�$70� �㻒7��j��ztʽ�Q
=��=�����>f�=X�:��_c��y����<<՜�Ư����<���=a��=D �<ꍽ���8\��'���#�<X�<H$�<P!?�j'=�Qj���:���>@t_�<��<\�=B��=�Q=e�-=ڲ����=�n�� -�;��h��G<B��=�9{��L��V8=�����x��6�=r�=H��Cs߽��K�C=a=`�h��>W�@И�]�><r���5=$9<�Z��oWr=�p������0 =@.;vO�=�)<�0���x�<�X�<��<b��=�l��'�����H<�����,b�Q���:L�=0=X�k<����P�\��� ������e <|ٲ�ܚ��`n<�;=�>�Ĭ��m��_Բ��1�;(u|�������P\=v�.��,=�v�=��_�@���u�=��<\��<BZ�@a��ތ�KFt=���<�l�pkм�켼�^�<��ɻ��2=�2�<�_x�dڢ=����	n�xDg<N=X ����<�v-=������Y��_='� � <|.ϼ�$㼍Y���*���X�;ƕ�|)��(*:�a!��J���R=�D����g=h�	��=�E� 6N��a��TG�<@�ݺX�.�b~ �,r	=P��teL=P��;�n�<@�/��;:�P�X< ;^�X���K=��-��Ud<�<؍=�LJ<@�g��m��ջl���1�0�H�|�փ7� :�=ȝ�<�P�;`���.� �Ѽ�lV���7<��<h�u=���<��ټH��R&?��9=��#=x�S�H��<Tv�<��1=1�½���4v�� ��;`qI=(�<<Ƥ=��J�4y)����=��=(��Hѽ<0�������Z�<�q��bp= �D<^W_�b�=��3��D>=��X<p��xk���;�0�4<P9�<�p"5��
"=@g:;pn�L��<(=����< �q��[�<́��O�����*��߿<Ȍ�AC=\T�=���<�m�<��<����$��=��=qT�0�� M���Ǽ�F����&��s �R��=h�ֽ���=�g�=�$�H�����<�&�;�ݼ+k�ЌZ<R˰=�ݑ=@���Ja����o���<�<d�,�B=0'��]<ܕ�L�=��W�>�=��=�f>���=��< �<%��=��
=��U�^�^=\ q��,�<�L� 帹�7=��H<��Zh=��;�f$����=z�U= 1��ݑ��|��<���=  +���L< w�<Mn>�T�� ��;XU�<�ַ�#��=�����Q'���<@k;��U=�qC�`�W��e<M/M=� �<�ҏ=�5Q��b�4��<�ɗ����᤽�"=t��<R0\=-<���:U�Ƚ��9���2����<�~��xv< '�<�d��Da��n�j�*����=t)<�c�2b��D2�<��*����n��= ��:P&E<���=R�=�b!=��4�&�<�/��⟀=�n�<�E���=X/<j-= ��9>݆=�0=��6���A=�׀��&==(qL�O`=02}<�4�;0��;c��%���Æ=���x�
<U���%�s9ǽ� Z���\��'N���O��-<�A�<G��C=a��,�\=���<�w=]�Tc����?��V�<�q�:��p���F�X9�<�{��V`�=��<� ;��1;Ѓ���{�<��Լ�Q���*=�Y�ϻ�M]���2=�\�;����2�]�p��;��X�@��Lr��T��K<����>���<����I�����<xZ<��?h��Xm<�<��=rN%=.���f���1��ӫ=8-i=�j5�,��<�@<��=����@e��8��P��`�8=x�z<�ج= ����j)��3Z=�-{=��5;�!7=���A~�� Z�<�+ɼ�i={&<�'��$!=�Jۼ�
$=�C�= �@<�3��,f�<�x5<��%�̦!���Q=���;8�9����<��?� �)=X�O<�hW=����၇��9���f=�H��yK�=�s=Lb�<��=��p=�	W�b"�=Ӌ/=�kӻ0#��$�$���ڻ��M�4�<�N��H=� A�x5ɼ�S=x�� 4�:(��?1<\5¼�;I�X��<ND=���<WY�����@6;��=�N�|��=�ΰ��d��pΧ<�<:|��~�=Ï;��O��L�<P-�<,�缪�_=�n<����.�<���^��=�>����w�<�Q��
��vNY=��;�<\j�=ʁ=h�D�pQ!���Z=��=t������:=��=ȩ��lk7��ƻh�H=��=����HJ����7��6м� �<�I��Ā�8a��Ę�=p��;�`<�8�#2��ȍ�<l��<p1��(���{V�J��P�k=P�;�`Q=�>?��/��M��v_=_e��@0�;�덼��^*�=h	�쏘���o=+�0=�=�~셽�W��oM���U����</�2=`i=lj�<�V=��;��[�Z3m=��e��:�<@ô�����h�=t��< ����\u�<>7=��ػ'�� p�:/(�=�%0�mU�=�.= r(��fj�>�ǽ���@��=N��`����^�6G������G��@���oy��1�0=�~*=��y�R�=�2�����=�l=6=�=���烽`` ����<��;y�<�놽\j=_�^�=(%�<�֙�ж�;,,ü��<�#?���u�X�=4�����ܻ�뷻D;{=`Su;X��: ���<�b������-��r�'��ǽ>><>RjF=�#9�y��h�=\���Z,���<0�b<!��=أi=��T�P_μh׼f'�=��=<�3Ef=��;L��<x����;`5c��,�x��<
=��=xWi<������a=��=��<w�g=�d.���ý%�<������&=�<`P˼ea=IC��s=V+�=��I��v*=�/�w�= ��;��}�th]��t=0t<�������<�抽��=�8=`o�=�Ӽw����{/=�K���C>��=��<�8=ʓ�=`�;���=�c=�I���<�%g�@0��淼� �<��<��E�� l��	��xB=�ϼ<s��H[�`U�;.�M<��8�2<䜢<Џ�;k���pv� ~z�<�s��= 7Ž�	W� ��:6|�
��EM=�_�l��<H�L�$+�<�1����<���HB< ���0(���T�=з����~�P�F<*��$��� �l<�ꗼО�<Lޟ=V@=���]`=$=@���xdڼ�p��Hi=fZ��B<V����
�tԆ=*��=�Fi�D�%�@����ּ��<��_��c��6�`� ^�=LW��d���ϼ�Hս$?�<�9�<P4�������ň�$)��P��<�~"��=L���T0����J�=+D����0˼�ba�r/�=��;�v<�GA=-HR=��Ҽ,�Q�T1<meʽ��V����;t��<8eZ<|��߹^=�!���?�O�=�\ּ&�&���h��o$��7a=�h<��3��t���Q�\H�<z�$�]G�l�&=s[S=Xb�T'�=�G= �H�dkݼ����t�=*�i�H��|��6�O��j'��� ��t�:A葽V?���)�=���<�����U�=�Q��9��=�ݘ=h��=���r���3:���<��<��4=���{U=8O9����=�=@��:��i:�J�m�<�W}�Wh��8z�<�|ƽ0�c�`m�:�=����D�(�z�� ��<A*ŽNA�����j��i�s�L>j�j=�������<�
����]z< �<_X�=b�~=����
 1��zs���=���= ��b*�= V��F�;��,��qB<pQ�4~\��(<99%=3->��g<p��z*�=f�=l��<x~a=�#5�MԽl��<$Pټ�H.= �O9`����j=��¼ę�=��	>@g;�?=�X��?7=�!�B�����c���p=��H< U�����:�ǯ��]0<8��=�˛=�����}�x1���J=�Ƽ��8>�=�>�<
*=�
�= o-�?�>!�S=l͵��o�<l@_� �;p6�;X�:� l���揽`vG<$�O���=豳�%	��Ey�� m0:B�<�_������H�<<��;��׼,H��Bֻ�v��0�;��x<<����w<Բ��ժ��Y)�*�,��Y�<�sʼ4ť<�2��0��@I;@�[�`^\;��=3�=�3ý±�� �o<�:0�0ƹ��U�,�0��/����-=�d�=D�	���=@!�|!������2���e=/���0(ӻ ����3�<v�=�G5=�)�(m�D��<PH�;�t
=
Sm�_��Ub��
0�=�@�8�μ��F�՞佘��<���/=<�������z<�2F��k�$��<ht;��*@�I���]?��ֵ�>r7���M��%�<�Y=��� ��; �9�@=��<T0��Z,=�Ÿ� .;@��:����(+��.�@�'=�}��҂;��y<�u:�%�����<蛽`��<Bu������)��뜽��<w������=,� ��}��n�=�O�<t����#�������N1="#n�l�N����0���b��`$��m<�x�G���Ŝ=,}<!9߽!��=�\��o�O=�%j=3ߚ=ȏ���X�`� ��d����=�=���/�4=� �Bc�=fb=D����F���� � G�9��J��͂��]F<��u�*t� �Ǻ�)�=�j0��좼��4����;�_���j��1%�i���j��"�>h�d=��h�2����'=���� �@~K<�K�;\��=�=;.ý��\�����=�C�=���ȱ�<?���h�4��0�ռxzؼ��PV����:���=p������Pqa=s17=�U�<��=����@�ǻV�>��E�<p��%����<`*��X=�o> 2<H.3<����	�K=<�������t5Q��B1=l)��J�� ����I��&����=�֗=��A��m��'���%/=�_��>	!�= ;���A2<�~.=������=|h�<�V���;��y�@��;��r<h��֖=�^̽�)�<`���p?�<D��Uh������(�"��]�W���m*�0ϐ;�a;0��;H�0�H�`�H���H��<L?��<�n�<��_�}ؽj�\��y߽Xc�gI<Xζ� ���@a�;�خ�������X�\��pm= o=T^�~����R<�2��䮼cԈ��uY���`� O�;a�=b�Uc�=��!�8��m��Rǽ+_=/Qý\�	�6ϓ�R�S=@�<@��:�`}��j��E�<f�<�C�<˄����������Wc=YH�<�Ҽ����ڽ�:!;x����< Р�닽��=����`�;8"\�T c����P�2Cg�/ނ��茻7�:=(fA�a�`�>�W �@�<<x
#=T|�#b/=����x1��į��iD�i���}��(�T<f� �{���`-�<����l)�<vjĽ𧑻ZqW��A��R�\�ҽpW����ݽj�V��=�1����{<�2�<P���Z�,�p�3�>����J���Y<�qh�����0� ���c�ƽ�#6��5�;�\�Xp�D��= �(<��Ы�;�A���L�:T�<�L,=�S�b���+������=�K<Vq����<�a"�=@<$&�<d�[��5/� ����x˼j��@*�'�����m���-��g6= _"� ����_�x"�$��� �r�8�2�f��v3�Q�=�� =��O�ޢ�B�=p��vI���x<���؊�<���A׽ �߼zZ�E�=���:�ռ ��\��ٮ:wA�!�����V�(�Լ�$Ｄ�J����=H�Ԇe����<`F�<�� 1�:CH��<� Z��1ד�/<��>�q<�<��iM�@���rT�=���<����D�H�4=��"�(��|�9����<�	H�(@�������@���޼B2�=+p�= C�:Џ���ļkS=�w!�v��=��=0�Ƽ���r� ��^�=0�!��+8�hР�XL#�2����u�T�j��l�<�.�=������=R��=��ƼX鄼1�= ��`eĻf�2� �;*F�= �=f#=Ǆ� ����M���ϼ �j:@���`~^��d�>=��<����<���=�`���e=�3û��O��=�,�<<,��p�V<r�콒��$��<��`�N=@��:;�����=���<���k=@%�<p�W�'�Н�;�j�=�%� ������P�>�(ٽD�<��?�T�ŽsO�=�a �`�%��*�H���=:jR���;0pH=p(=@`:;&�=Llf�eˑ�z*��8t��!�b�i�9��=��<�L,=�4�<��h�t���(_� ��;�:f�����-�0����R���UD��&���۽ �	� E��,���Ͻa=Rjj�HV�'= 64��e�<��=�kK<��4={��� ⼂-��d�<= p��&��`��;�m».J�=�1#���k=p&�;�"��Gw=8��pF�<|�<h)<��:���<�.=���N�_��~k=z����:�ژ�������Q������؆(��|J<�c���I��vW=�Iҽ��S=pu���%I=R�A��л�0��� �G<x�+<�k*��`���R=DX�1�Y=�j<��#= �'�LԼ��<<q�<�J��8a`=|!���W<�q=j�=(��<����E[�(/�г������W�t0����=����=��J=Pٔ��߰� ����f��jpd��R�<�A����t=$��<�4:�N�9�z	f�'!�=��A=8Bq�e�B=N�<$j�<�V�0O�$]�����#nz=p��<C��=,����'��e�=���=�r̺�=`�+�����y�0=�ˠ�s�$=p{�<-ႽpǾ<l����a=�ƒ;@���	����; �?<P��;زJ�x�<� �q=�`R<�����O�;LU���"� T/:k�7=���}��$�J�&=t���/�=��=L�=r�=�ħ<�r����>��=�m�VC���}e��D��~��T<�0���>�����<h�=n� �<qg=�x8�Z�� J��{8��g�=�)<=x<�ƽP��;p'h<����<S�<����q�<�<��=�W7��5>P�<P������=T�ͼ�������=x�<��d����:��� �E���;<p\�L>=У<�UE�-��=T��=�=���=и5���&����(��=�ڮ=�ɼ� �һX	��{/�=������C�p��;(�����=��� O���1=��ٽ��s%=����pu�Fv=��<L�;c>=�C���s�Ht�;�������d�=�w�;8��=H��<��=e@� @A���Ҽ�gv=b*�.�#�����'��2�<�~���e��hQJ<8�<@��Bj��p�ּ���
���K�<���<D�r=�n�=�=V�=�;����k< ���x��=��z��T�·=�}1=x��=��}�N��=�3�\�<��=p1��B�=�9��i�;�� � X<@��<�aF��ڍ��~=,,�����:P���h�>��P���'��p����lG��)<֍��SS�mg=O���p�1=�SZ�CL=҉�@3������ЍN<��t<�^ּ:|��d#=�岼$�6= z�<���<��M<@-�h�=@˷:�BM��1j=�?�@�:��=�Z=��< o����7���;�K⼨����"����E�p,�=BSu=b@�M���-<0�o���Q��g�<@��:��B=x��<B�P�F���l9�*ֹ=�k4=���=|=���<Ӳ߽[Ǽp����T3;	(U=�x<ī�=�v��$s���si=rY�=po�;M�=D4��錥�8V�<>5���?=0 <.�!�h]�<tL��q"=�*a=�f�`��; (9 N�<�<8������JŇ=��W<�����;�"���<8k�<�n=̷��ߕ��g��j=�rĖ=P�p=X��<0S�<��==�+�;RO>$��<X�< �^�)���B�<<,o����=`��#
>8Ƚ7��0�=<���ğf=h!<�}���m�%c=Ј�h�n<��&;6�x�4�P��<��<X�)���:=�Y����O�a=���;�֢���U>B?���r���**=�E�a�����=X������:�㹽̨���h�<�������x��<�l���KԼ�W�=�l=�+<��=L���<=ټ�~:�E�=��= ���7���һ�Y�(��<�j?�ئ���W�=��=C��7�e�����P�"�Kɚ����}�=��=��Y��tǻ4ql������\���=��#д� C2�ȿ�#=����>f�m�X�R��h���H�=='��	-����,��~^�3�=P�ռΖ!�d=$�<l�_���=�2;����Ƚ�˝�����=�r=���=��.�T�=E�B=���*��=�dJ�[�*=��p;��,��=��=�^.=/D���s<��.�<�<`�j��2>�I����<�{�:��e<�L�< .g�ˬ� �i=8���ቼ��0W��]G�� [��Ă��|���q�=�=�2	<)���0�=O���+w= n�<t=lS��m����P��;@k�<�ր;~��Q�D=Hg���7_=1=`�<���<�����Z=�b�3N���z=x�=����;�8=��@=�UC<�����{༔s�<NH��ļA���#@�[҄��K>�=�s�����j)<D�����޹ =���<
��=��<'�����E�����{�=�g=�	[��6=H�=0��<C`���I`���<�J ����<��<ʞ�= �޻(�<���^=-�=�KH<ġ�=�#��!����Mh<�W+���^= a���z�����< ��(YN=*��=mQ<�== ���<�/�<x
H��v��=��<�u ��n'���h����<��=^��=lӈ�q�����
��3�=�C�����="%�=�U�<褱<��=@�<�>P�<��;��ʒ�����<�����
�=��!�'G=Lht���:��<v;.��D�<֟3�d���u`��x�=��z��/;��+o�u����ʽ�q=s<�]:�Գ�<��ὶ���/=j#&���f�:�/>Pe޽0:E�pbջ�������f�
=P��\��<�1ý<X̽��Z=�n���ѽ`V<4k��;���JK=���<��;��<BM��LD¼���<��=x c�8{-���0;��	��0v=^˓��s#��V>�=Sլ��%e���������#���@�W�p���Y��= �����pC�3F� VN:��=�ܽ�⩽Z~��T���<�
��Y�>HBr��x`�wڽ`A�=q亽�۟�d�缶�#��>pȏ���n�$S�<��<L�<���������W׽6=[����T��<P�=�߽Iߊ= 
6;��ȃ�=0���t�޼��hV���=:�u=�����t̽0���<��|v��{���A=F�>�W��4=@��|�<Hp<6�ѽ6��V'=F�>��>P�\kg����	���O���D�X����瞽%m=dו������=�X��dN�=$�K=V�d=��ڼBl�@���B6����<��<��n� �=����G�='�n=�=�' < o"���b=t���튒���?=<�}��ߟ:�fo=:Hn=������"��������<�3����!��hI�Q����>��s7>��=	����罰r�Jz���L��<Hʻ<B��=��<6ֽ�߅��5C����=ύ= �l�VD=���<�}@<rW"� 绺p2�;��0��D����<<3> �!�����%��=�y�=�d�<t@�=D�M�Ƚ�0��IV��S=�b��A��a=��Q�`W�=���=��<tH=8h��v�=p�<�\Ž �-�9K�=�-<����� �,ҩ��R<��=�>�=0X��������_�
נ=��C���">�Q�=`��;0�j<cϡ=��;��>4R�<0%b��9�X	t���<\��<�Ό=ʵ=�z�;�MѼ����;��q��넫�8�.�ϙ��'�h$"� 2;�硼��=��3��d$=����p���h%Y���׼��Խ�7;+�����):<.m=���\9μd�(��P��ɷ����������!�<7�����@����=����H�h{$<r�B��(��8���t+�h�ü�� ˻8F
�V�*=�F=�N��^(��/��F[<ʜ�<=<���`�h����=��<-_��k��Pu����ν@����L�};�n�>�I��=t�{��5>��ު��� ��;�W�<t��{6�y'�� �:t�׼�H���[�=@ʺXq�J��~�=�f�$�� �[���D����=`彻 ,X;����D��< T���м�����\���ȁ�`~����&�JB*=*�ӽ�҂=�P'�]V���f�=ā=���ǽt"��Q=�=��.:�d���������u�s���T�˼��=B&=P�һ��=@%�� �� ~����ڽ�>Žp�<��H�R㜽,����D�w�޽x���'�<DB��A����= S����wt�=d1��d��=h�a=�N;=������g�8�|�R�P��hH=\i�<��E��p�=��Ƽ@�=�4k=@��:�}������!�<���ne�H`�<н����[�E=��=�]���Y�x8ȼ���;�&����K��so�O齽�c�(��=jƜ=����ؽ�
�<��
��R7��E<ԣ�<Q�X= ��:$_�VLC�0����}�=�/X=�6�<P_�;�$ݻ�����J�=���Q;@  ��^Լp4���>����\�W�x��=��<<�<��j=�!(�;E�� 4��ƛ��"=����(u�<�0�;��X� 9�<0>���< ��7�WV���4=6�;��ὸ-�k��=�D� ۵� �T�H���T�7�>��=��H�?N����d��y=���tq>��=,����m;Ӱ<=8�����>0[�;ټP�P��Y��ͮ8=2�=�=�2x=�9ڼ@�z�@�ݽ�������7��j̽��7#���.��������;設��!�����h%�<�����Q���2�,u�<���M��A����.��!"������S��ă5��3��Ô�d=¼�zb����;����lF���=Л��,��(�)<��S�(o��b�2�ߵ�t�%�v�M��J�<�;�P%=�<����o
�������<�1 ����<������<L��=`�<>���� �t��� P���G��/���k�Jv�=��q�X�Z�xRJ���� �Y���
�`0<��?�ý���<��C������=`<���fh��䳽��׻V�+�?m���6<�3�<�{Y=���`Ц��J����< c�<LN��h��<�o���j���м����81����8�?=�jD��z��(b�<s,q=����鍽�.����0=0ȼ�y�����@<н|a׼�D{3����=�U�����< '<؈�8�ż̛����ɽc��pD�;&a��o��.蚽Л�<OAý�1�`�<P�׼��R�\V�=d�ڼ�K���=L������<�'�<�1=pQ�� A�;�qּ'��;�q=��;�)�V(^=�r���8S���#=����6� �ػ��a��b��
'� ,p9 �<:�J��,*=(�`= f��8`;� ���û���4n��b�����,9�i$�=�%}=�m��'��|=ȱY�t��8�n< 7:���<�ؼ�y�� ��)��E�=0�<`��;8t����߼ ��>��]���4�	�K�V���h����=�"��u��E=�$"<�B<�<gm������bM��ɽ��<��P;�<@� �������#����=(�<^�p�ͻ&=��Z�"	ҽ�1���u=^c	�����P?����x�����=#I�=@W��q��HYt�BB=LJ�����=B�=$D� ���Y�9 ���>()���m
� ̽���@C��������t�Ӽ@v>8,����J= ��=j%���;�a>=T�(b���8<��
��H!=���<e==�D��`u̼h��<�ٳ��aѼ@1.�~�C��Cv<��I=~) �=�B=�36=�ڮ���0=n��)d�,kd=��<���zp)���W����������l�<p$<Õ�����=D��=4�4��ʻlM�`�B���:�=O��=xh� >�� ~����=n`��`Y;��o������L=~
�$����K�Q����=¹~��K=.�=�s���R�<��%=�Ns;�.���w��T�5�H�y�g�І�=�������<Ȟ�<�u=�����D�@!.���=�|���hN�"��4�ؼ �ʼ��o�o񪽀���T�=t���{T�P:<�\���2��@�0��[.<`�7=Iʅ=8�6�=�Ϯ��+:�q��H��<xȼ�1� u�l=ba�=߈���x�=RdF��]=/�r=l�N�'�}=�6U;������<�c�<߁.=P�a�D�/l1=:����<T���D��<�9Խ,�� 1���ϼ$�	����<bY�h�μoo=u�Ͻ�G= ��;:�[=�!���üH�+��'����;�޼B
M��8�= Uм�\k=��<�CP=@��T��lO�<�=�Zc�sE=�������<�TC=�v�<�-=@ <�L�������<��� ���8"&��f��b�=JS=�d�s�ĽPfN�8�ۼz�0��Z�<�{��Mu=ȪA<J�/��)"���E�ER�=�+=
O�v��=��<�ɕ<7!Խ��N;zS����t��=Ԋ�<V��=d�����:�8.{=���=({.���<@T������=�s��r�=(+�<��v�X7�<Nc����i=�q��!:do��`��;P��;�e��C���A;� �=��=�p�l�<4R-��&#�P�PCi=؇ۼ�;����Ӽ=,=�, �qt�=�#�=�4%=��<(�:<L\��C>���< �{9N弽����pYY<��%�=`D����<>򮯽�%t���O=*(Z�i3=*��=����C�E��=.:U��)��P��; �m:vz罐u����#=�����h��
���}~ؽ�=�4=�e�|h3>�D���Ͻn=�o�`t�m�=X�1� ;�:�.ͽ5����R��R��`
�\q�<���<������=�'�= ��;��<�p��|��B����
>�̰= �,��h+��<���;�<@��Оڼ�Է�x�<��=l� �F���� 4����;����h�~<<�=�!��
�<@�1��:�9���.gp�^�=w���`��.C�=\�K��,�=�p�<��!>mE��|���t��J�=BsH���v��]�o掽��M=��6���� 7K;�=HcE�D2
�4M������p���༖�=_��=��;�<s��=��a>C=�怽��=\�@�E�9�5=>��=���=y�Ͻit�=�.��"n�= [=�}z��A>�0���{��qh<���<���<�⸼�`��.=��N���";@�\G�<��r���R�\����\�����?�<�z���ƭ�e{+=c妽t�
=�$�;'=�6�1q��H�(�� �;�v�@�¼4�.=Зݻ��=%�<ԯ=�#H<��w���2=���<� ��EC=�II��A�;�F=�s<j�	=�I�:�O���,<`��;�Q��8����<b�,���Nh�=�Q�=XM�Z�t�PeĻ���"�
���<��+���= �;��o������=^^	=\?��K�d=��< r�<����� ��l�� >��K�R=|��<j;�=�\��xVü@2�<T��=�0Ǻ��~=H�[��H��H=\"����:=�͒<<�ۼ�4�<�}C�3<=(�=x�S<(�b<�y<p�<@���`���p� ��a�=:=q#�h]K<�y޼櫼0�<��y=���~E_�0�^�B�f=p�m��k�=8��=���<��<�=<�$�=H��<��<�T���7����=諒-a>�B��4�7>��ĤW��z�;�y��j�=X��<J���!��GV�=�7��<{�x~
���������9�<�/=(^C��XL<73�f�@�(�=�W�<��;؍>����Ž؆�<�I����G��F�=2i��w{=���|�PE��t�Ҽ�
�� vT;`�<(�<�z�=Ao�=�f=�M;��H���޻�^H�9�+>��<(j2�X���P������;ڌ=�=���|���>�sp=�2�V�����~�r��q�V��د���C�=ܘ!� $o��8I���D���˽J�/�r�>�N�IC������NԷ����=�ī��Y>ya��`�&�L���?�!>�L˽W���fNq�'�ν��>p�Ի&���<���<Nx���]G�B(���25���A���={`&>���ԉb=ͱ`=V�H����=L9��E�=��R��&=/��=N�>5m_=r���W�<	��e��=`;(<�9����>����t�Hl���<~�=�.Y��C ����< �˻�Ơ�X{�h�<t:���;.<)���;@���8~=c<6�}Iv=����Z/-=���<� =m�����;%� ����͍;x�'<����u"=��;(�<sJ5=Ь�<�C�<p�����=xx7<8ȯ��g=� Ἰ�%<&z�=�<� �<(�q�8<�&�<�S �`ʌ; ��k��j���
�=§=S����>��݂�lG4�H)j�EHD==e<h�<P�D�����T<� J��$�=�M�< �`<�OH=Gb;=D��</Q���ᵻ��A<��p��<���<*S�=��|���'���K<3��=@��:��=�z��~e��?�<ް�7�m=@��:�V�:@=�żw�7=�=�=�=#=�%2<|k�<@<<p���ر����=�F�<���Xv��1��`P��Ιv=P��=�{���0�x�b�*�=��;���=��Z=���<�0��
i=�*=F3�= �I<T5�<���N���/=`
��->h-)����=�������d�׼UL���3[=��T(�QS���p>�����:��h,�[;Ͻ��׽�K}=�ȴ;�Ix�ȂA<�d��R��$�=09�LK�<�<t>�B�`ϕ�	���=H��@�pY�<ďM��N�= H�{
�ȯ~<��!��_ݽ�O����f�<���=n�~=xt�<�����(�  �`%�;�	>�0���K�
���+J�TaB�f��=2����1Ž�A4>`��<oͽ��r�7����s^��,���|��H#i��
=-�<T1F�4�����5���쳼5">��-��ț�AH�����=3��4>�τ�����	��>����2��޷	��/���(>�Γ<�N�:�=��&;��{� �"��%ֽ93���ẽ�N���M=�&>�h7�I��=�
b<h�5���>�~[��E�;��]���<�u�="e�=�!<����ڼ�㛽�7;< K����y<�qg><������������=@��<��B����Y�;Ǽzz��_h���2�FZ��K$;�`���*�����X= ��㐾��=b/a�ûy=_�9=Hw�<8fp�xQ��O8��0+� �<8��<p��b= �;	=Ƅ�=��<��<4���͉=x���]�9=ؐ����;���=d��<  �:Ȥ�إR<Xa�<��&� ���P���n�/�&�N�&��=k�=��ý�⇽4��No������.=d�<�x>=��ż�5������`*<�d�=h��<t4�<�G1=��=���<]Dٽ�F3;d��<��޼@>ں��t<���=0��,���,�<�[�=���;^��=p�ļ�c��Y��WU��`V=h�A��؝<� =4(�n�T=���=_L=d��<�V��(��<HEn<BC������7F�=�3�<�)�<a#��Lt�X�8��l�=�Qw=H߼j�b�,���ۋ=�{S<D�="��=�	�p_�����=&x=.�=�uH;X�<�����f�'�<x�=<���=�!�<7_0=�3���K��ż�낽�lA<CZ������l�z�=�D�"] �dd=�sB��Vg��xe�=�T�@�,� `���ue�F!��<ޖd��L9=��=D����������.����� ׻�&�a9=:���
���-=�Z������Լv�� &t9��]<�OL<4̍�&�o�
x���u_���<}�=�pW� �X��`��6��f2�l;�= t���a��<�>������l�$j�F{��K�6��B� �D�>��巼��=^����T���m������g�:J4�=�e��$���5��d�=���̼�3��/��=8��<�����ҽ��o=N�a�����H�[�2N�V�
>\"�<`lU<�|�<`�R<hs�����$Qż�BȽ����.������=�{"�+S�=:o ��޽��=�ǰ<��ƽao�$ּp��=�m�<T@；���8�|���E�O%��p񱻢��=ӓ�=`�T��낼h���5�;hC�0�[��Fg��PD���¼ dýlכ�Ԋ�<�@����@�<pf��Ă���|�=0�9��T��5�@=xz�^�%=8�j=|�< ���l��<�ݠ�'ۗ�?=�,�<���6�i=� �: �?;�΂=`�� » �R;�Z�<�B!�hn����c<�C=<|;��_�e=f-<=@�X� 6����;phe<bp�|��0�'�Z�q�4���
7=C��=+½&�K�@p�;X`ܼ �};���<���<(q<��;�}س��X� ꚺ�D�=��<8�R=`�<� ���@@;9a½���0E�<p�ܼ�����弉U�=����(����<L��<8�n<
T[=:��=閽$Cj��맽S�=��a;٪{= r�;�:	��D<J�>�L/=@��� -���=f*<d�����μ�8�= ����~�4)i� ��|�üx�>nYx=�Z���SC����<=�6<<�=ƭ[=�ͼp��/=�S�;�f�=�}���4�;d+��0an�6�"=�R�<%x=�-=1�<�g�,U%����ʎ�0Ή�5����2C�F�i���==��ƻ0�6�t�V�}j��4R�K�o=|���/��zȼȐX�����@@,;]z��df0=Ui=���@Nܼ(;����z����^z��V�i�=qa���{�A=�*��B��LoҼ�8_� �=�`0� (���#�a7���b���ټ�6O<�q=�Ck�(�C�،'��ch���%��;=�x��T��/�=0�6��c� ρ�vt@�!���L�����H� �6##���=ơ���?��@���������P�<� FV���X��v�+���ƽxq�=�Tj<@��r��`��<Jr��ϙ���Ѻ�iǼ�L�=ج�<`�v;X�[�$;�<p�;�'i����;7���ټX�%��.l��T=л��'c=�@H��謽�Ǖ=�/=����c۽p�$4A=�<��/6����:����zO���Ā��DJ�=h��<H�}<�S� �7�
>�0����;�������ԙ��<���`���	�)=X����2��'?=��B�0���V=*��O5�� �>� t<�'Y;���<0ݻ;𽅻q�C=\���νlV= ��9����%= ���� �;�=\��@%��_�<H_��?�;�	�:��`4=�� �}0=�
= ���ܠ�<�!��K��P�ϻx�����=���k�(� <�<<�RR=����z5�&T =���;�{׻X$3<���;Ȑ��@���z��8A���Ǽ��0=*:�y�=�B���Ǽ@����7���W��@ö�@��(L��fȇ�-��=:�&�ؚ<|�<�V뻰��;xw�<�|>��r�����=�Ľ�͓< ��;W�=t_���M��G򼫠�=�S�<\�#�.�; = d�9���ͼ���=t�ȼ@h,��Q��O<��ӼKy�=�Iq=�/��^�?����=�<���X�#= T=�X���ؼ����l����&�=�������n�̯�� �c�����yN<�����=���8��<���<n�� O=����<���(��=��y��웼 ̓9:~&=� ��~�a��	=�� ���LR=��/���N=D��<R�`���5<�ƕ<�`-� %�9�f�L㼘�U<�Ć<@�r<�g����=���L>(���3<�~<ȯp<g+��m=�p�=h<�]*��都T1�<geֽv��=�;�=�\��K�֥���}�= �b� I;�҉<p�v���r<?�̽��.���z��'���T=]#��14L=�=4�>�|.	=pWK<��=0���������5��\���O�=܃ ���!���X:ѹr=/H޽�,�`-6;l��<�Z'� rI9Tݼ�\��nom���`��� �z���F=>�x��� Y�;6�+���Խ�P2�NF=��=���<DC��� <�����/�<���=d󎼀*<����y=�=n`�"{�=eI��z�Z=)�=�ys��gQ=���Hz'<3:'=O�5=:�[=h�;�����_b=Ĉ����X<l"ƼǠ2=�� �ռ�]��<M���q���c>=�@K� �� �<CY���]C=��J<(vW=r������ �i�(fS���;pZ�~��rώ=hDq�@�v=���<�g=��e:�3��=bY*=�!�`�?=އ��<��<�e=��X;�]4= `�����:�K,�g�G=�)����f�@�M;�I{�*y�=�H=VZ�[�ŽPQּ���+Ҽ�[	=$벼�n=����rO�|�ͼ,��,��=Z3=�1"�'��=0Ǥ<�*�<}�����<Lϥ�,���9�=��<K��=x�'�\�I�?	Z=�|�=�D0�8/\<�-t<�i��r�=hfx�XG=6=�kK��=(i����f=��o����<�|U��<P��;0]a��#'�H}<�Ŗ=�$=`��;<��<���>|=���&��k=�N|�{х�������<����$�{=`��=G�:=��W<Pg�;)���S>^�<���;@GV����$<�u����=�L�[|$>\5��28�� �T9Bo����<>�8=��G�>�|�_��=O|��;x�����@���@$��vA�ȦS= �g��&μ4���Ƚ��=8��<ҷ&�AX>�[j�p�{�_�<ƦX���۽̚�<8���as=��Ͻ�н��� E�@B#� d<4=�< �����=��=�i:=���e�&���<���0;>{}\=�T.�`s��$x���}�:��y;4�F*:�A�<x|.<U#����6�BZ�l[:�0ǆ�������<ҭ�=�e何b =��ȩ�<V,X�/鍽l^�<O��G���p=�z��r�9=�V��r>cz�p�~^���Q�=�R`�~R�>_C��Q��x�z<��	��#��Y��=,v�Ш���:��=r�����7<�N��=(��=�J�T ����=.���܌=��A��w�=4#���ni=�n��8��=;�=��½�[�=�����a�=��=�����s<>��� ����h=O=���<�Sټ0�e�h��<H�����;:�U(�w=*%t�h'C�\Ս� jһ�����{=X�	�`�;<�<_O���= ��<
�=P�ʼ=ؼ�,�dyƼ�����J<ȣu�ga_=�V<;�w*=�r=��)=8�y<Z �^D@=lN�<�t=�<C�Ђ�<	�k=�ǈ:��=�A
��"�<���<���<�R>;L� � ��<�¼�3�=��~=j�f�����Q���N��^y���
=�|��-v<=@ܼ�,��^��@L8����=�v=8��=���<L��<_�ߜ<̕-� ��:J8=Ӡ=�D�=�r!�T{#��ې<ַ�= Ҽ�;=02�;���c�=lG��H�(=�F�<d葼r�#=��(��$K=���<h��<�c<@��;��<0鮻���7�;���=i�4= �E9��<�����1�H�#<L=4O��ʌ"����Ks=��K��l�=���=�`�<@�;��=��<]w�=�Ȣ<t��<@T漪����M�<hgʽ.�>��->(��<qm���6������!?=df�<��K�a���Zd>ހi�����K��m�����ǽ�Fo<	�D=t-���;�)E���&���>\�<@X����s>�!	��d����9;U���$�+�jT=�w����=6��j�	�*	�H#��0^�����D�<C�=.��=��= g=�,μ[�c�(�<~h�t�3>���� ��';�ں�]�轚�=��tS���>��;#�X?���H��Qg�k���;��T.�����=����@<�F���ʷ����2vt�q>na��Eg� '�9�3����=l����E>X�0[ݼ���>Y�/
����d����5�=���;@)2�ϧ=�;�ٜ��[�;P��O뽺[��x��+l�=�<&>^�&���j;T�=P�>�h�> -A�9^�=^i1���=��+<�\)>X ?=T��h�<Ѝ��S�=@ku�P���^��>�Eཬ,ü�K�:�=P
=`��h�z���;�=�|���h� ���<LO���7'<|���@�{<�軆	= �?<pk3�=��e���<w�<�o<��^�hSV�XA!��#�0&����< 7;�=<��<���<e�+=�<�a�<�ۊ�$\�=�T�<� t�m4+=蘺���<��=�Kp�$S�<htj�X�=� =l��<8��<��� U�:��Ǽ��]=��=g:����%��`�P�Q��慻�:=4G�<h��<���t��z�(����<	)�= G9<$�<Agk=?7'=C�<BA���s<��ǻ@(ݻ�ޕ<��<��T=�NU�,/�Hς�Y$�=`D�K�w=pt=�@f���<=��QG=`!�;@��;1�-=�C���*=Q݇=��U=���< �D<,�<�L�;X�ȼ�7i:b�=�
=���;`�0< _��������0=�w=x�N����:��E�U=\#�<�W=�Td= d)<����LA=ښ.=��f=�' <��<�*��T�T�<p�X�h|>6Z�����=d�ʼ
d��띇��V����=p�3��l��r�<�P� >���N�`���������R=�s�;|�d�8H|<m�.�§4���=,o�<�2�<�5X>���\�"�̈��K��&����<0U���=V@�Z^� 􀹠ѼH���n-��إ;FA=�Ss=��{=Zf=�n�I_=�2=��!<���=s"��1��=�T�����$�;_�=dxQ�N0��+#>P~5�����)�f #��iI��/��J _���Bs$=��R��@'�2����j�������;���9>QZ:��A���x�x���=��f���>��"<�裻s 
�B��=�ֽv<|�bf���ݽ�>
N=h�<M$u=l:��׎��ں;p�ٽ�ܽ1e��n�����=��*> �T�N�	=��N;R��>xd�P`x=�4�&�=�/�<���= r�92���r���Gֽ�0'=�S,�Dݽ��$j>�Q��\ȼ@����	=�L�<����:�4ݪ�X��jW��R1�03<K�����;D��h�<HC[��
0=��;V�W��L-=D�8�=�y(=pd�;d��#���a3�:/f��p��Z�< 6-��]=@~y<���;�f=x��<X�<�J���<�=�l6:p�5��&=��� �l<i��=�K;pY�;���n=0?�< �.��[T< Ha��X�f����P=�Y�=�Ӿ�f[F���X�n?�� ��?%3=�A�<��<�2��v�Hb��(�<޲= 8�;JK=k�;=*$=���<��t� �Y<���;<+Ѽ;���}1<��h=�# �|�����ʼG��=@w;��M~=�����:I�`CQ;��*�F�"=�޻PQ�<��=�H��x-=�x�=�̀=xr�<�$<���< \<\Z:� jm��G�=�<h�L�`�q��K.�^���1�=�c^=�8��J������Вg=d��<��l=��f=p�V��=8�|ln=�(=�mb= 'c�dx�<@tq;\�X����; C�:��=L����%=�E����b���`������A<�U�� ��@����=��"�՝����r�r!���/��$�=��#�(|)�P`<�Ƚ��&�R=��ѻ<=C^�=*�۽`Tۻ�̒�2�L��8�8�h�0����=�󤽾6����<(�~����A�� �ȼĺ�<��<�-<����<;�Nǽ��<��<.�e=H�U���L���콰	��
�T��=���D�]�=�!���5=�d������  �-L�̨ȼ\	��0����=�*��2����Tٻ]bŽ���R�=rc��"�eν�߮�d�������K=�S,=@\X;~W�1DU=o;��r��:<����(|>��D=p�=�P=�k�{�8����̼u���XH���!��`]Z<}8�=r�4�F==��}����8�=���N���齘O�<L�<���<F���_��P�K�������ؼp�9��V+=y��=���E���=-<��W�9м���R{���s��ެ�P�|�hn�<�?޻����0f�;�D���^��~F=��������q<,Є�tĻ<Bmc=���� �1: �k;pvl�yݟ� {6;8��<�$L�T6=Є&<�n6�;T= >� ^Q9@`���=@�ܺ��>;���;�F�;P�	�i�N=���<���;�}C����<$�<�8� h�;8�����a���q�̘�<�DZ=�ɽ�8������� ���lS�<h�k<�w�;�Ќ��Q�z�	��[-<،�=��;�3I= ��; �P���;�]� ����:�<BܼH3}��4���q=X= �HO4��徼��O=<��E=B޼�0�@��{��4��< 
�;B�u=��\<t��xJ<7��=�N=@���Ծ;���<�!<;��m��ﺼ��=@>�;���Xe��e׼��(�@$�=�M=4:��Ή� ����S=@ك<�9=;98=f�����J=q|<��t=�j�H*d< �;�>S�`�J<X��<�lN= 9���^�<����:�.;V��ʝ��;�#��h�)� �D�=^�=@ƺn\
��:��+��hJ�Ŭ�=�k����� [+;��j���ܽ��=��ǼޡN=˃=�������ј�v�'�c3佸#T�P�׻5zs=������@�+;���&����\��^��@��:xX��E�;�����p���r������<�q�<$h���<�a���|�T�~O:=ؙ^�j���O��=!ؼ��ͼ��&�����ɽ��� |��8��DS��n�S=b���>|��P��;��˽����d�d=�i�`�<9Ľ �2�~)(�í�DB�<n�= Y�Z��\(�<�~!��8�����,w�p��=�#=`��<�y<�"�:�<���m��A0;�^� �������#��=*���p<�<V�:��M~�8��=�@[;�;��>֙��E�:��*<����:����$t��?��V�/����<u�=�,= ົ�^��Q!�@�R��m�� iM��tD��./�pB��j򭽔Q\�C=�< z��n�<pA�ح�<L�<05�;�x��^��S�<8���=�됼���;�]%=p�߼�Iǽ�Ϸ< N79��);0$< �;�H9����<��!��v��X=��`�;,=�I���E=D:��<P�<�7�; (�<H�W<`�%�X�u<xI<��Hֻ`T�<�����<���p �(�<t�<�U���-< Q{;&�������I#�6�;SZ�xa�<�Lؼ��+=��P-��,�:�����&7�@�G<�a�*��fi���<N���)�;D��� ��9$���`�O;n�!��ּ�k��g�� �8H<D*�=܆��h�����"�=�ζ<H,��7<�U�<@2��p��xJü=�n=�W�0!|�������<����K�=7g =���|[ż�����~�;8%<��� }�:·�*��P�#���wC)=䄄�Xw<
�
Const_2Const*��
value��B���"��T���`����<,eʽ̚&�(���_�=ą3�0k�=HDZ=���?e����l���ٳ����;�ā�P*	<M@�=��/=�$���`;��,=��J�T����?<<$߼�3'=�ߒ��,O�Da��|]k=�T���X���<��<l�ͼ¶�=��V=^4�=�T�HT<Z]���M&<$0�=�?�<hg�� =���<82��T�� �N;�Sd=�˽�xH�=*QN=�?o��H�,��O'�=7���Z�<��=�0=X�<e�
ȍ�+r<J�J�P��;!ҹ�A�<��=��ȼ4 =v:R=�X.�����vJ=ЛS�8W��"׽��=l��<R���i�Оv=�^̽rւ���"�@g^<h�p�`>�<H��=\:��d����l�.%5����$r��K����r�=�~=r7�xϼ]͈=8���P�=������=�&�<m��P��=�&���=@��;��a;r>2�vw9=��=�b�Ԣ�=ҏ�B >���>(y�<�M=P����̻4U0�0�,<�ue=������h�n<���M�&=�5�VY�@v�;xkM<x�q��r�<�]��b��4
ļ-+� �<�$Ļ �H<�y��Օ����r=��F��%:=��»���0<t�0�\��~����0��*�;�:���(<�u�; Pd�d����Z�<�1=@�B<X�A<�Xf�`����m������5���::П(<��ڻ �;���<�𼸪Q= ����ŻLM��D����,��pR�< �:���؏`<��_<����"K�X-㼌�I���８��<}�'=@��� �ڻp[L<h5<d��<�X���9ͻP�<���;���<��м�5	��BԻvG/��5�������)�(�_��	C�B+-� ��; �мȢ���$a�ZՖ���9���p���GD=�"�<�ර������Ӽ(�I<�=����m�=���L�˼pu�;²4�������� ��;�^r���*���3�pd�D��8�D��<���<��y� ��Ks<qz=������7�:���hS�=`�߻��=^(�=y^<�Z��8���3��|���`�;
v�X�<�X=l��<JK����t��+�=��!��r�� 4��. =���<��:<�G�@�<�-=@���(��x��@VK��7��>~=l�=�Ų=�i|;���<4d��(!�<g�}=��=�I�:�.=6yx=A�x=�=��=�3�=����ڳ=	(r=���< �8�^�;\T�= P� 'y:�d\=���<�_|=׻νL��<�<$ۻ��S<��m� �&=C�< 5��(Td=�b�<`�<2�w��E=��*�����`��j$=xM�< l�;�	�<,�=����7;У�����<Ys�� �9=�=����	t��{[�䷸� �9��:�3��8����a=��������m=(�P<��B��O�;N��=�N=8�¼]�=����6=��==��<(׼ϔ�=8��<P��<KD�=��`'c;���=@�;�ê=إ�p8�<�ш�����P~ƻ�n��L3���:�<���5�>=�P���+���(�����L5����X;�	�A,� �=�;�ܼx�R���3༤a�< k�8�<P ܻ@��<���<5�`�z��·��,��gD<�kC�`/��8v5���<���E��b]�`(����;\Ƽ(������; @��pl��'Z�w���S�ȗZ���<@�L��"���0U=X�ɼ�c5=����Z=(U�<fI���ɻ0V�;(�&�:< 
\;'�#=�^�pA� �`�D�`�NǼpC����:=�P_�l�д,<@ٳ�9�<W��]�<|'�<��	<�L�;V���׾� ��j�9���.��>��3Ǽ �k����:�	ټH)���qȻ@��� �.7�ぽ������� 6�9��<`��<�ؼPd'���p�;�j=@*t<H��=����Ӣ;p؁;�uż�I��Rü ���X�ݼ��  �� ��90ñ��,�<��<���IW=dI�<��F= f���
�;H��H���6=}�A=��=H��<�؟����=\���T9��T����m�<�%}<(0�$�K���S=8-
�!�$=p$���=�~���=8����m	=8p�<8f=�n{��>�0ŀ��� �೭<���=�F�=�Æ=(!=f�D��<��)<�c= n�;���<���<��
>Q��=C=jR�=r	=�=Q'#=h")< Q<0�=�su=�x]��o��F<Q�<RI�=K�Ͻ"����:A< <|<�ŏ<�2}���==�¿� �E�= ��N=�哼�$=P��<��<��K�Hi�� 뼈�<�"=��.=� �Cs^=ԁz� b?�8���I�=��&;�>�<@��;�h=H��<��l= ��<4�d����;�g1=l��� ������<��=`7'���;ʟ�=��`����<��=`Y<H;�<�r�=�:<� ���e=b��A�6=��t=��h��<�����;����=�x%��L=�y�<\~������`&��R����=е���3=pc���ۼ����Ϫ��)�X"q�d| ����:@��=��<�%�^����|� �:~�o=��ڻ@��������<���:t���y9��c�������=$�C�lS�@ri�l��<��*�V�'�Ѓ�����h�缚�"���Y�<�Wh��X�;���@�z�nP$�����Ђ�;��D���{��4�=��<:V@=��d�fz=�=V3(���<@���P^�;4��<��˻��A=�����< �f�����vĻ�t��V�K=��м��˼ ���}��\��<�m��s�.=@�;@Ԩ���)<�u*�`�����;�ۼ�8K�D��<𲢼�RF�5<X�b��0�@��;T⥼|9=���@�:x�Q�|��<�����.�<`�ʼ e�:��� ��:�Bt=�:�<(O=����<0�	<��L��+|<PA��.����<@_<��<p��;`k;�x�%�=��=,���$�f=����h<�V�8c0���ٽ��ǽ��o= *r;|P�=ৱ;(<�Y����==�=�][���< �<����I��(6\<��F���޺�<4�^ �=(�g�oܨ=꺓��=@n亀�úȣ�<���=�N��H�N<0��;8�T�p&H���4=={2=���=���< �<;t|�<A޼��<HN<@~�ɶ�g��=M��=)�=H�#=��=𹷼��C<��H�J�o�H=�6=���@c!����8C����X=9~��,lo�Б�;P]�<T��< D�:ʫ =��}�h'<�r&�`�����:<�{q;���<��
=��e=�ڟ�f�n��s��%<�R�<X����;춈=Z��v��6A�19]=����xh�<�=}jt=��G=/&�=s�:=�0�H�k���!=������<8W
<��2=(h�@���t��<����U=7�=@徺@b3<��=����\f���z:�3���{H=@���<��V;����e��r&<(E�Q�=�{�<L�����ּ�W�<����^�<�l.��L�< �~��3�Ԟ�d�Ǽ����xiL��&��=Z�=@��:��8�����0������;RS�=��E;P�O<�c���a껀�y< F�@�ջ�ч�@�(�t��<�W�����^�:�#=�é��d༈�6���D������;��c˼�X&=��:�l<pc��@&��Ӷ��dT����; �:��`�	�=��<��X=����tԸ<�O;=K��\=�ϻ`�y;���<����"=<����=��C���D��JI���k�b�Y=�t����еM�H2(�b�;��&�e=� ����:��#<��h<@�;��;����6L���=�j����;ؕK< ;������a<���}=��&; �d�PC�<4�/=@��xv�<0���� �;�@�h�U<��3= >=РL<0��G=\d�<�>��=@�<.j �Ք=���<\�0=�)�p��|��b'?=�Z�<�ݴ��Z�<�cļ�ؼ�O�(�i� �V�n��4�< �;��= r(����;�8��0��<���=˩���S=��<@�2<>hH��� =2l��J� �T�$��=�����=ܠ�5Y<8�;�_ێ��1X=F3n= �����6=�Ee=�C �����P	Ҽo�F=<��=������<�|�<�� ��������<`Ѽ���0+�;�=��=���q��=�H� ߀��Н�@,�ιC=\�(=nq�ܸ�����#_�|�<�� ��x��k[<j4=`=t�����л��}���;�����A<�"�0�Q<�\�<�(�G�< �7��B'�����	; &��HC˽�tj<|h�<]$�����N���P�<����`�<���;�9׺lX�<f0@=��s<< ���D�,�<����	=��=��=�U�@�m<4+��M��L�j=���*�� d�;Rx�= 3�_ڊ�O��&�y�� w=x�ۼe(5=Y��@�m� �T�\�L�x�	� �9�}l�Ȗ���N*�t��<�����"]< Q�����<@�6��� ;��j�,2ݼ<���d������=r70=d�ϼ��9�J��`
$��=��Ai'=�Qb<��=���r�� ��<04� ��:nԼ�d�p���L��0����1�Y=�nļ`<��pË��랻�?H���>���+� ��<`tx�0�j<(ݼ�+�� ���!p���; է;E�<�I=dT�<��i=�������;"&=Ec����< ;������P"�P�Y�h'�<T��&o<`��x�#�T�#�h4x�m�x=`���ʻ>��V��`�h<���:,gz��;=8{��h![<H� *R9 �:��軆�
���I�<�vL��;@��;�+!<"v��@��:\���0= �: ��tE=%+=̊����<�+�;�,����)����<���<��.=p��; -v;�
=��<l�ּd�<\S�<Y� K�`s�<x��<�L"�~��s7���b=،.��]��ѐ��
�#G��L0��࣡�hhݼ��ݽ�����5:���<l�ݼ��ռFN'�d���0�H=����h�= j�����<��ļTN�< �8�X���h0=�nF<�ֻ�xK=�0A�|:��*N#����8�3=\z
=$��.@==��=�r
���x��r���'=��L=v��Pq<_C<J���zP��j_<~�VV��
u����<��Q=�ML����<�򏽸�0_黀pU���=�]=�Ľ�Ƽ�4�ĝ���`��:�< ���,;}l=E<b�*�\r��x��p邻t=g��ݨ<<O�@-�:���<�k4�d�;�����ܼ`�������^�������8�輬ͼ'�Ƚ�V�;P�/<Fn��ܽ�^@��8��&�����>��P&D�\.���?�<��.����<5B=�;x V���y<�)����6���T=�E�����x{]�jQy= �z�k7��!5ŽvW4�Wde=�����<*@)�HY㼰�;n�Ľ $r:Ƶ^�X����5���6��,�<�G!�8�r�� T�9��ݼh�<��/<�]� ���{���^9���6<l��<V�L�CΜ��d�8����}���_�8<D/e=�w��d��<�����,<!��m�4�f����t���5���=^�_��o���ڝ;PP}���?<陏����� A)�x�>�8�<�K��-�0�<�¸��@��:'�"=<�輀[S;�7=�Շ��ڋ�=M7����<�2���d�8�_����p�=<��ʼ��@�`����1�vr���Z��]=P��;{C��d��`t=0}��%ٽ��3<*$��=.���P;��p��(���2{�
�8��qX���6���[��s���#�<��ν�s������|{�<Xpz�,N����<�q�<���x%�<������ļ탨��= �{:iqa=��4��<�A�<��z<r�O��< +;�4D�2��̣ͼP��.<x�\©��ᖽ�5=�,I�J2S��E��P�M< ���,ڽ����@�����=�,���>�= m=���}6��ri�(��F�;�~7������=��=hh�<��@o���W=�.���m<�\�:�W�;���<�5<Rvl�޳~���=�|���S=lڏ=��R<�8Z=��F����=d�M��D�<������<��=�����+�T8�<��׼����p��<�U=�&�<����_�3=�]=P�
��˦;���N�>Iڽ)�f=4��<�đ�D��<Cꬽ<����8=�WQ��̹<�0s��4���1<�W��I�=\q�=d]d�I����B=��f��8����Zđ=��|<�<�Ƥ+��X#��YŽ0A��@N�����;�Z���%=��=��ż��½�3��^�I���(���r�RMǽu.=���~n<�8�=���� <N;Px<d�=�׏;�	���r=�Z�x�a=(@�<�^�:�.a�0��<�*�<z�;�Vy=��<����1�=�dW<�:�;�Y��h��<htǼ,��<�r/=`D���1����<�\����=4似N������ ��; ��7p��84&��V��z��F�`Sg<tJ��,r�<케+6;����([<`5�s1=�^�h#O��Uջ@��:I���< _Լ�M< @Ը��;PՓ;�!���y��P��<hU"<@�!<�) ;��;@Pa;��лR�N�H�0�`���p�;�|�;�<�<���c=Ȑ��P�4=0Z��~Ի�*��T���I�@}R<E�<��;���<��@<Ă�����`�u&� 5�̏�<^�0=P�� 3<@�ź��<5= 2�<ے<���;�V���<�-u<��̼`�ڻ�����ȼP�����Y��RK; d���%�hԝ<`�M�H<주�y����HW��3/��~�<(Ck< 0�:��غ����͆;_�=0~S�#�[=pj��p �� 0�9b�G�H��� �z���b<�I������ j�;�D[;�k^;0v��6=�=�!����;xI�<X��<�벽���l�b��L�=�����=�~n=��i:��0�4� �K;j/<�"fR��ȣ�Y7=�d�=�?޻d����&��d�=�@w��!=�̫���$=H'�<��<�L>�@َ�O�=���X�g�l��<�
=���<��L=P�<�n�=Й��m�-=
C��䕻<h�i=�s ��@��6c=���;���<��=7l�=�X=r� �k�T=MN=Ht<
�=h�<�>6�����<�=��<�p�=����
�?�&=�ޣ<8p�<��F���2< ����Q<<�=G=��m=����yω���^=h�������wĽ�p= ,<��<�.u;��<����8���f��L�<���z�_=ڝh=�����4�8���,)����=��g<���j{��{=TӼX򊼪>�=tɀ<����`F=��|=��<h�<s?�=\=�UD=��t=�+�@na�P%=��J<�=�(�=@��< �g��l= ��:"J=�T�v�r= ��< ���8w��ao��%'���9=�p����<8��
�@�F3J��� ��J�*�L5�PxI��>= �������L/�lV�<�A_<Ɂ =���6�`tZ��=�G���iR�\�H_%�T�ټjR=@x�������Bǻ���;`��Z�/��3Ǽ��;��������:�#�L��<�m�:� ��;-� L2���6�`DC����;�a&�R�"�!��=�r˼��<��5���=�<�./��4� �S9)�=�|�<���;��=Է���<`��<�o<��<@�8���=v�C���ݺ�l�:����l�	=�_":zo=(N}<��ȼ�s<��u<�]<��
�;�h��QT���<P_���3;J:<�.���<�
M;�!<��.=Hmp� Q��ؑ�ȗf<�\<@��:�� "<`��;8X��C�j=��K�WQ=h�8�<�ݻjG���b:85���T�0��<��L< �<�9�<�n=�M�G=;�H= �p�Ň=��1;��;أ��`���Zڽ���@��<��1=��= ���3¼D��`�<�쀼�R������^M=��<�Nx���J���U�=`8鼗��=���I�y=�0;܏�<�h�LY�<��<=�!=V��hR�<%�X�Y<l��<B!=��=�[=5�=�dl�0Nd<l'�<`�0� �y����< A�9F�=4�=���=K1=u�+=���<ty=�	<	�<P="�= �p��5����;Ġ�<zE�=���P�3���=�}=@ԭ<��%����<Ƙ���#=��< ��`C�;i#����]=���<x>H<����n��p�鼂#=h<T<��<:,�@��:��W�ܟ<���"�=�y�;�G��/V=&�=�=<~A�=��4=��w��2��&=��?� �`� K�<� =���� 0�<ߨ�=D�<�d=�=�3$�hT1<�h�= �ܻ0|�;('�<��q�6=(JK=H\= �Q�P�w� q�:��b= aD��Q�=�&�=�0v���A�Nꏽ�0��O�n=T�ļ��<���:�hc�7#��T���x�kP���r�<T�<dV�=�0û`4;��~�v>=m�_=�=�n�㌎� �;q�=����`�<Rkq� tƻ������=Xzc�x��������W� ��;D��vf��k;$f���7�B�r��1=�BG�+�l����<���8V�`v�;PH���9���X>0w��<�}����2=(�U<�4� �u�������y=�M8=0o�����<xt����=>XY=��Z�!'=ؿӼ� =Ͻ��\�;@mѻ��.�Ц�<H�<0X�=���<�t���<�?=�+�;( x<���;I��@=p/߻(x<���<L��,�<p֨<��<��= ���=ȣA�Ac=�+g�8����(�(°<�^<�����p=<Ϫ����< W��)�W=��L��������<����C�����=�I==��<.A=��= (���Y=��=�}j��7^=�'׼����3�te��l[�P����3K= &�9(=�=8Q�^g�Q/��l�<�O<����i,<�+�< �;�5���ʼ�&��z�< 	�zֱ=,����̤=*-�@������ ���[<;��=�a߼"%=��w�@胻 �W;�c�<|�=���=���=��S�p �� c�;x�V��0s�PR3�d��T5�=��=���=�<o��=����%F<��P�8�� �=��=�Ȼ�˧��g��@G�;:��=�?����m��N�<��(=�=^'����<1ܗ�(�.=�sv�`��0�;e���M9N= �F<��7=�W�I����O:���=`�u��ܼ`i��0(><4]��h��h�)�:�H=��4� ���BȌ=�*=�=9�=��q=��x�C��$N=�j�� T ;�` <G	=�0� �:�K=�?��*�=jC=��뼌^�.~�=x弰�P���r�魙��V#=��D�̉=h��ȓ����<��<8�����=o�=���ȚS�>Ǧ������Re=&���;ֻ��b���a�m콬���56��}��◐��bu=TÔ=XX���}�<a$���$=e}�=H��=0ð��姽�W�;�<H�]��<=)�����;�N<��1�= �#��/׼`����ڼ��-;A؋��\Q� D�;�D���1�֩T�4�b=�{>��R�����`��<?���P�@K��A�喦�?�%>0�X< ]
��l��K�=���;�G�pF�;XD�Om�=�!i=��ۼ���;pY�祯="�=0jW�i�P=�ؼ�s�<3�ɽ0�)<l���gc� �g<�]�<�o�=���<�8��D �<
Ռ=tY�<�ڠ<�s�;���mb+=���x��<P�&< ���<h�<�ց<;�=t��=�E���!=\�0��EH=���(�"������<��#<�oM�I�&= �� '��`��;֓z=�%������<�C<ȷּ�r
>���=T4�<l�<qс=��}�=��_=�ʒ��h
=�0-� Ţ�� üh��,�i�+��(=�<�=�q���M�Ǐ��xv<�>�<4���v�< �`:�A<�{'��#��Ƙ �p�ɼ �;��P=ĸ#��~�=Rɒ�4��N7'���ý�l�<V�~=�5��2<]=���<5��B�Pr�d�a=Q�=hE=0WL� ��� }S��Z��`����.��F��P�1<�=�K�=`����ƴ=P�&�p6���컸e'�giz=�<�;\�������F<�k��y�<^�0��`_��'=nЊ=��=Z'K��<�����<8����'��w��l�E� =� ��"=�߼��|�0���t<��мX�g� �ι !лx�}�R�2���� �L<N-Y���s<@$�<p̄�x.�<,=N=���;�c���]=����l��<�9�<�c<ި��pԮ;�::<g+��<��=x�����/���<�y0�=t�s��,�����#젽�=X�r�
=�8N�Ƴ��	=�tT�`�ڻ��=`d0=�c缔�=����e��E�=�O�a���2���?�н4�=�`fe�E-�����j\�=lZS=�C?��Y<���5�<�x =f�=p��J*v����>���<�\�<q*��࣋��4�m��=���:tk�������K� a`�Bh���9����:�=���5.��^���U=�k��xr7�4���]X;66N��9����\9���ek����=���<�Dw: ��C�<���;�\��;�;�E�G>=�\=N�-��?���s����=@ =��)� n�<��鼀ͼ<�e������ ���� 1����ջV&�= f��^I����b<e<I=���< ��;8����ڗ���<���1|< =��㻐����;�(�<�J�=����U/<�j���$T=�!&�l&��f޼�h<�湼�)���<X�ż@�D�P*�<|�i=������Q�<�c�<fG�W�=*�=h�~< ����h�<�	r��y�=xE�<�Y��'�<TiC�b9�`���D��HV.�b���Ϛ<�F<�}�<ģ���S��(��`Cb�PF�;4��p�<X���aP<�߻p��;�:μ�1���=Pw��S����=����y���q�N�*�$��<��="��0=�^o=�?*��B ��z�D�<A��=pD(<��k�Ht����޷G� �&��D����������z=��=��S���i=������𯬻8"ӼfnQ=$k�������}?����<��C��p����9<�n�b�=�]�=��d<V�f���������x��Tj1�@v;��ּ�/�h�z<�G��Xu�<��4�j�7��"q<hܰ��?]����0dӻܳ!��x;�ᠧ��I:��Z���t�P��<.�H��@r� Q��ؐ��xI/�$w�<H!i��w$=�y ��w~<؟�<8$T��Ƚ ~�;­�A}����`=꥽P!W�b;z�n��=�����Q���榽!,��d��<6�����<����0��Ru=��߽`,[;t��<`�p��\0� �)���^���� u�;�^�$̊����ԏ��j}q��FE�8����et��hV�GN=��=M��ƹ ����ػ����R����< ��8� �����)R��Oj;H"k��D:�(мH4� �:@�U��zw���7����<Zp�vK�|V����I�$!�+���,f����<T���p����@����dڼp&��w��d¼�1��P�g=��H<�ZٺW����<h<��0J�;�������:�[����u�a�DG���=t�ּ��.�@
�d@*��ն<R"v��ۂ���3���<��"˼XFv�� :=`�μ�=�Y����&<H�c�������j�����$�z�F�����દ� ;L�g�Ժ���V��[�=��(:�����/���=�p9�`����Ҽ�� ��π�x��@yĺ��;L0���<�O=��;Z�u�P<*<lJ�<��=�� =Hi=������-�DG�\P%��u=��|�W]�$70� �㻒7��j��ztʽ�Q
=��=�����>f�=X�:��_c��y����<<՜�Ư����<���=a��=D �<ꍽ���8\��'���#�<X�<H$�<P!?�j'=�Qj���:���>@t_�<��<\�=B��=�Q=e�-=ڲ����=�n�� -�;��h��G<B��=�9{��L��V8=�����x��6�=r�=H��Cs߽��K�C=a=`�h��>W�@И�]�><r���5=$9<�Z��oWr=�p������0 =@.;vO�=�)<�0���x�<�X�<��<b��=�l��'�����H<�����,b�Q���:L�=0=X�k<����P�\��� ������e <|ٲ�ܚ��`n<�;=�>�Ĭ��m��_Բ��1�;(u|�������P\=v�.��,=�v�=��_�@���u�=��<\��<BZ�@a��ތ�KFt=���<�l�pkм�켼�^�<��ɻ��2=�2�<�_x�dڢ=����	n�xDg<N=X ����<�v-=������Y��_='� � <|.ϼ�$㼍Y���*���X�;ƕ�|)��(*:�a!��J���R=�D����g=h�	��=�E� 6N��a��TG�<@�ݺX�.�b~ �,r	=P��teL=P��;�n�<@�/��;:�P�X< ;^�X���K=��-��Ud<�<؍=�LJ<@�g��m��ջl���1�0�H�|�փ7� :�=ȝ�<�P�;`���.� �Ѽ�lV���7<��<h�u=���<��ټH��R&?��9=��#=x�S�H��<Tv�<��1=1�½���4v�� ��;`qI=(�<<Ƥ=��J�4y)����=��=(��Hѽ<0�������Z�<�q��bp= �D<^W_�b�=��3��D>=��X<p��xk���;�0�4<P9�<�p"5��
"=@g:;pn�L��<(=����< �q��[�<́��O�����*��߿<Ȍ�AC=\T�=���<�m�<��<����$��=��=qT�0�� M���Ǽ�F����&��s �R��=h�ֽ���=�g�=�$�H�����<�&�;�ݼ+k�ЌZ<R˰=�ݑ=@���Ja����o���<�<d�,�B=0'��]<ܕ�L�=��W�>�=��=�f>���=��< �<%��=��
=��U�^�^=\ q��,�<�L� 帹�7=��H<��Zh=��;�f$����=z�U= 1��ݑ��|��<���=  +���L< w�<Mn>�T�� ��;XU�<�ַ�#��=�����Q'���<@k;��U=�qC�`�W��e<M/M=� �<�ҏ=�5Q��b�4��<�ɗ����᤽�"=t��<R0\=-<���:U�Ƚ��9���2����<�~��xv< '�<�d��Da��n�j�*����=t)<�c�2b��D2�<��*����n��= ��:P&E<���=R�=�b!=��4�&�<�/��⟀=�n�<�E���=X/<j-= ��9>݆=�0=��6���A=�׀��&==(qL�O`=02}<�4�;0��;c��%���Æ=���x�
<U���%�s9ǽ� Z���\��'N���O��-<�A�<G��C=a��,�\=���<�w=]�Tc����?��V�<�q�:��p���F�X9�<�{��V`�=��<� ;��1;Ѓ���{�<��Լ�Q���*=�Y�ϻ�M]���2=�\�;����2�]�p��;��X�@��Lr��T��K<����>���<����I�����<xZ<��?h��Xm<�<��=rN%=.���f���1��ӫ=8-i=�j5�,��<�@<��=����@e��8��P��`�8=x�z<�ج= ����j)��3Z=�-{=��5;�!7=���A~�� Z�<�+ɼ�i={&<�'��$!=�Jۼ�
$=�C�= �@<�3��,f�<�x5<��%�̦!���Q=���;8�9����<��?� �)=X�O<�hW=����၇��9���f=�H��yK�=�s=Lb�<��=��p=�	W�b"�=Ӌ/=�kӻ0#��$�$���ڻ��M�4�<�N��H=� A�x5ɼ�S=x�� 4�:(��?1<\5¼�;I�X��<ND=���<WY�����@6;��=�N�|��=�ΰ��d��pΧ<�<:|��~�=Ï;��O��L�<P-�<,�缪�_=�n<����.�<���^��=�>����w�<�Q��
��vNY=��;�<\j�=ʁ=h�D�pQ!���Z=��=t������:=��=ȩ��lk7��ƻh�H=��=����HJ����7��6м� �<�I��Ā�8a��Ę�=p��;�`<�8�#2��ȍ�<l��<p1��(���{V�J��P�k=P�;�`Q=�>?��/��M��v_=_e��@0�;�덼��^*�=h	�쏘���o=+�0=�=�~셽�W��oM���U����</�2=`i=lj�<�V=��;��[�Z3m=��e��:�<@ô�����h�=t��< ����\u�<>7=��ػ'�� p�:/(�=�%0�mU�=�.= r(��fj�>�ǽ���@��=N��`����^�6G������G��@���oy��1�0=�~*=��y�R�=�2�����=�l=6=�=���烽`` ����<��;y�<�놽\j=_�^�=(%�<�֙�ж�;,,ü��<�#?���u�X�=4�����ܻ�뷻D;{=`Su;X��: ���<�b������-��r�'��ǽ>><>RjF=�#9�y��h�=\���Z,���<0�b<!��=أi=��T�P_μh׼f'�=��=<�3Ef=��;L��<x����;`5c��,�x��<
=��=xWi<������a=��=��<w�g=�d.���ý%�<������&=�<`P˼ea=IC��s=V+�=��I��v*=�/�w�= ��;��}�th]��t=0t<�������<�抽��=�8=`o�=�Ӽw����{/=�K���C>��=��<�8=ʓ�=`�;���=�c=�I���<�%g�@0��淼� �<��<��E�� l��	��xB=�ϼ<s��H[�`U�;.�M<��8�2<䜢<Џ�;k���pv� ~z�<�s��= 7Ž�	W� ��:6|�
��EM=�_�l��<H�L�$+�<�1����<���HB< ���0(���T�=з����~�P�F<*��$��� �l<�ꗼО�<Lޟ=V@=���]`=$=@���xdڼ�p��Hi=fZ��B<V����
�tԆ=*��=�Fi�D�%�@����ּ��<��_��c��6�`� ^�=LW��d���ϼ�Hս$?�<�9�<P4�������ň�$)��P��<�~"��=L���T0����J�=+D����0˼�ba�r/�=��;�v<�GA=-HR=��Ҽ,�Q�T1<meʽ��V����;t��<8eZ<|��߹^=�!���?�O�=�\ּ&�&���h��o$��7a=�h<��3��t���Q�\H�<z�$�]G�l�&=s[S=Xb�T'�=�G= �H�dkݼ����t�=*�i�H��|��6�O��j'��� ��t�:A葽V?���)�=���<�����U�=�Q��9��=�ݘ=h��=���r���3:���<��<��4=���{U=8O9����=�=@��:��i:�J�m�<�W}�Wh��8z�<�|ƽ0�c�`m�:�=����D�(�z�� ��<A*ŽNA�����j��i�s�L>j�j=�������<�
����]z< �<_X�=b�~=����
 1��zs���=���= ��b*�= V��F�;��,��qB<pQ�4~\��(<99%=3->��g<p��z*�=f�=l��<x~a=�#5�MԽl��<$Pټ�H.= �O9`����j=��¼ę�=��	>@g;�?=�X��?7=�!�B�����c���p=��H< U�����:�ǯ��]0<8��=�˛=�����}�x1���J=�Ƽ��8>�=�>�<
*=�
�= o-�?�>!�S=l͵��o�<l@_� �;p6�;X�:� l���揽`vG<$�O���=豳�%	��Ey�� m0:B�<�_������H�<<��;��׼,H��Bֻ�v��0�;��x<<����w<Բ��ժ��Y)�*�,��Y�<�sʼ4ť<�2��0��@I;@�[�`^\;��=3�=�3ý±�� �o<�:0�0ƹ��U�,�0��/����-=�d�=D�	���=@!�|!������2���e=/���0(ӻ ����3�<v�=�G5=�)�(m�D��<PH�;�t
=
Sm�_��Ub��
0�=�@�8�μ��F�՞佘��<���/=<�������z<�2F��k�$��<ht;��*@�I���]?��ֵ�>r7���M��%�<�Y=��� ��; �9�@=��<T0��Z,=�Ÿ� .;@��:����(+��.�@�'=�}��҂;��y<�u:�%�����<蛽`��<Bu������)��뜽��<w������=,� ��}��n�=�O�<t����#�������N1="#n�l�N����0���b��`$��m<�x�G���Ŝ=,}<!9߽!��=�\��o�O=�%j=3ߚ=ȏ���X�`� ��d����=�=���/�4=� �Bc�=fb=D����F���� � G�9��J��͂��]F<��u�*t� �Ǻ�)�=�j0��좼��4����;�_���j��1%�i���j��"�>h�d=��h�2����'=���� �@~K<�K�;\��=�=;.ý��\�����=�C�=���ȱ�<?���h�4��0�ռxzؼ��PV����:���=p������Pqa=s17=�U�<��=����@�ǻV�>��E�<p��%����<`*��X=�o> 2<H.3<����	�K=<�������t5Q��B1=l)��J�� ����I��&����=�֗=��A��m��'���%/=�_��>	!�= ;���A2<�~.=������=|h�<�V���;��y�@��;��r<h��֖=�^̽�)�<`���p?�<D��Uh������(�"��]�W���m*�0ϐ;�a;0��;H�0�H�`�H���H��<L?��<�n�<��_�}ؽj�\��y߽Xc�gI<Xζ� ���@a�;�خ�������X�\��pm= o=T^�~����R<�2��䮼cԈ��uY���`� O�;a�=b�Uc�=��!�8��m��Rǽ+_=/Qý\�	�6ϓ�R�S=@�<@��:�`}��j��E�<f�<�C�<˄����������Wc=YH�<�Ҽ����ڽ�:!;x����< Р�닽��=����`�;8"\�T c����P�2Cg�/ނ��茻7�:=(fA�a�`�>�W �@�<<x
#=T|�#b/=����x1��į��iD�i���}��(�T<f� �{���`-�<����l)�<vjĽ𧑻ZqW��A��R�\�ҽpW����ݽj�V��=�1����{<�2�<P���Z�,�p�3�>����J���Y<�qh�����0� ���c�ƽ�#6��5�;�\�Xp�D��= �(<��Ы�;�A���L�:T�<�L,=�S�b���+������=�K<Vq����<�a"�=@<$&�<d�[��5/� ����x˼j��@*�'�����m���-��g6= _"� ����_�x"�$��� �r�8�2�f��v3�Q�=�� =��O�ޢ�B�=p��vI���x<���؊�<���A׽ �߼zZ�E�=���:�ռ ��\��ٮ:wA�!�����V�(�Լ�$Ｄ�J����=H�Ԇe����<`F�<�� 1�:CH��<� Z��1ד�/<��>�q<�<��iM�@���rT�=���<����D�H�4=��"�(��|�9����<�	H�(@�������@���޼B2�=+p�= C�:Џ���ļkS=�w!�v��=��=0�Ƽ���r� ��^�=0�!��+8�hР�XL#�2����u�T�j��l�<�.�=������=R��=��ƼX鄼1�= ��`eĻf�2� �;*F�= �=f#=Ǆ� ����M���ϼ �j:@���`~^��d�>=��<����<���=�`���e=�3û��O��=�,�<<,��p�V<r�콒��$��<��`�N=@��:;�����=���<���k=@%�<p�W�'�Н�;�j�=�%� ������P�>�(ٽD�<��?�T�ŽsO�=�a �`�%��*�H���=:jR���;0pH=p(=@`:;&�=Llf�eˑ�z*��8t��!�b�i�9��=��<�L,=�4�<��h�t���(_� ��;�:f�����-�0����R���UD��&���۽ �	� E��,���Ͻa=Rjj�HV�'= 64��e�<��=�kK<��4={��� ⼂-��d�<= p��&��`��;�m».J�=�1#���k=p&�;�"��Gw=8��pF�<|�<h)<��:���<�.=���N�_��~k=z����:�ژ�������Q������؆(��|J<�c���I��vW=�Iҽ��S=pu���%I=R�A��л�0��� �G<x�+<�k*��`���R=DX�1�Y=�j<��#= �'�LԼ��<<q�<�J��8a`=|!���W<�q=j�=(��<����E[�(/�г������W�t0����=����=��J=Pٔ��߰� ����f��jpd��R�<�A����t=$��<�4:�N�9�z	f�'!�=��A=8Bq�e�B=N�<$j�<�V�0O�$]�����#nz=p��<C��=,����'��e�=���=�r̺�=`�+�����y�0=�ˠ�s�$=p{�<-ႽpǾ<l����a=�ƒ;@���	����; �?<P��;زJ�x�<� �q=�`R<�����O�;LU���"� T/:k�7=���}��$�J�&=t���/�=��=L�=r�=�ħ<�r����>��=�m�VC���}e��D��~��T<�0���>�����<h�=n� �<qg=�x8�Z�� J��{8��g�=�)<=x<�ƽP��;p'h<����<S�<����q�<�<��=�W7��5>P�<P������=T�ͼ�������=x�<��d����:��� �E���;<p\�L>=У<�UE�-��=T��=�=���=и5���&����(��=�ڮ=�ɼ� �һX	��{/�=������C�p��;(�����=��� O���1=��ٽ��s%=����pu�Fv=��<L�;c>=�C���s�Ht�;�������d�=�w�;8��=H��<��=e@� @A���Ҽ�gv=b*�.�#�����'��2�<�~���e��hQJ<8�<@��Bj��p�ּ���
���K�<���<D�r=�n�=�=V�=�;����k< ���x��=��z��T�·=�}1=x��=��}�N��=�3�\�<��=p1��B�=�9��i�;�� � X<@��<�aF��ڍ��~=,,�����:P���h�>��P���'��p����lG��)<֍��SS�mg=O���p�1=�SZ�CL=҉�@3������ЍN<��t<�^ּ:|��d#=�岼$�6= z�<���<��M<@-�h�=@˷:�BM��1j=�?�@�:��=�Z=��< o����7���;�K⼨����"����E�p,�=BSu=b@�M���-<0�o���Q��g�<@��:��B=x��<B�P�F���l9�*ֹ=�k4=���=|=���<Ӳ߽[Ǽp����T3;	(U=�x<ī�=�v��$s���si=rY�=po�;M�=D4��錥�8V�<>5���?=0 <.�!�h]�<tL��q"=�*a=�f�`��; (9 N�<�<8������JŇ=��W<�����;�"���<8k�<�n=̷��ߕ��g��j=�rĖ=P�p=X��<0S�<��==�+�;RO>$��<X�< �^�)���B�<<,o����=`��#
>8Ƚ7��0�=<���ğf=h!<�}���m�%c=Ј�h�n<��&;6�x�4�P��<��<X�)���:=�Y����O�a=���;�֢���U>B?���r���**=�E�a�����=X������:�㹽̨���h�<�������x��<�l���KԼ�W�=�l=�+<��=L���<=ټ�~:�E�=��= ���7���һ�Y�(��<�j?�ئ���W�=��=C��7�e�����P�"�Kɚ����}�=��=��Y��tǻ4ql������\���=��#д� C2�ȿ�#=����>f�m�X�R��h���H�=='��	-����,��~^�3�=P�ռΖ!�d=$�<l�_���=�2;����Ƚ�˝�����=�r=���=��.�T�=E�B=���*��=�dJ�[�*=��p;��,��=��=�^.=/D���s<��.�<�<`�j��2>�I����<�{�:��e<�L�< .g�ˬ� �i=8���ቼ��0W��]G�� [��Ă��|���q�=�=�2	<)���0�=O���+w= n�<t=lS��m����P��;@k�<�ր;~��Q�D=Hg���7_=1=`�<���<�����Z=�b�3N���z=x�=����;�8=��@=�UC<�����{༔s�<NH��ļA���#@�[҄��K>�=�s�����j)<D�����޹ =���<
��=��<'�����E�����{�=�g=�	[��6=H�=0��<C`���I`���<�J ����<��<ʞ�= �޻(�<���^=-�=�KH<ġ�=�#��!����Mh<�W+���^= a���z�����< ��(YN=*��=mQ<�== ���<�/�<x
H��v��=��<�u ��n'���h����<��=^��=lӈ�q�����
��3�=�C�����="%�=�U�<褱<��=@�<�>P�<��;��ʒ�����<�����
�=��!�'G=Lht���:��<v;.��D�<֟3�d���u`��x�=��z��/;��+o�u����ʽ�q=s<�]:�Գ�<��ὶ���/=j#&���f�:�/>Pe޽0:E�pbջ�������f�
=P��\��<�1ý<X̽��Z=�n���ѽ`V<4k��;���JK=���<��;��<BM��LD¼���<��=x c�8{-���0;��	��0v=^˓��s#��V>�=Sլ��%e���������#���@�W�p���Y��= �����pC�3F� VN:��=�ܽ�⩽Z~��T���<�
��Y�>HBr��x`�wڽ`A�=q亽�۟�d�缶�#��>pȏ���n�$S�<��<L�<���������W׽6=[����T��<P�=�߽Iߊ= 
6;��ȃ�=0���t�޼��hV���=:�u=�����t̽0���<��|v��{���A=F�>�W��4=@��|�<Hp<6�ѽ6��V'=F�>��>P�\kg����	���O���D�X����瞽%m=dו������=�X��dN�=$�K=V�d=��ڼBl�@���B6����<��<��n� �=����G�='�n=�=�' < o"���b=t���튒���?=<�}��ߟ:�fo=:Hn=������"��������<�3����!��hI�Q����>��s7>��=	����罰r�Jz���L��<Hʻ<B��=��<6ֽ�߅��5C����=ύ= �l�VD=���<�}@<rW"� 绺p2�;��0��D����<<3> �!�����%��=�y�=�d�<t@�=D�M�Ƚ�0��IV��S=�b��A��a=��Q�`W�=���=��<tH=8h��v�=p�<�\Ž �-�9K�=�-<����� �,ҩ��R<��=�>�=0X��������_�
נ=��C���">�Q�=`��;0�j<cϡ=��;��>4R�<0%b��9�X	t���<\��<�Ό=ʵ=�z�;�MѼ����;��q��넫�8�.�ϙ��'�h$"� 2;�硼��=��3��d$=����p���h%Y���׼��Խ�7;+�����):<.m=���\9μd�(��P��ɷ����������!�<7�����@����=����H�h{$<r�B��(��8���t+�h�ü�� ˻8F
�V�*=�F=�N��^(��/��F[<ʜ�<=<���`�h����=��<-_��k��Pu����ν@����L�};�n�>�I��=t�{��5>��ު��� ��;�W�<t��{6�y'�� �:t�׼�H���[�=@ʺXq�J��~�=�f�$�� �[���D����=`彻 ,X;����D��< T���м�����\���ȁ�`~����&�JB*=*�ӽ�҂=�P'�]V���f�=ā=���ǽt"��Q=�=��.:�d���������u�s���T�˼��=B&=P�һ��=@%�� �� ~����ڽ�>Žp�<��H�R㜽,����D�w�޽x���'�<DB��A����= S����wt�=d1��d��=h�a=�N;=������g�8�|�R�P��hH=\i�<��E��p�=��Ƽ@�=�4k=@��:�}������!�<���ne�H`�<н����[�E=��=�]���Y�x8ȼ���;�&����K��so�O齽�c�(��=jƜ=����ؽ�
�<��
��R7��E<ԣ�<Q�X= ��:$_�VLC�0����}�=�/X=�6�<P_�;�$ݻ�����J�=���Q;@  ��^Լp4���>����\�W�x��=��<<�<��j=�!(�;E�� 4��ƛ��"=����(u�<�0�;��X� 9�<0>���< ��7�WV���4=6�;��ὸ-�k��=�D� ۵� �T�H���T�7�>��=��H�?N����d��y=���tq>��=,����m;Ӱ<=8�����>0[�;ټP�P��Y��ͮ8=2�=�=�2x=�9ڼ@�z�@�ݽ�������7��j̽��7#���.��������;設��!�����h%�<�����Q���2�,u�<���M��A����.��!"������S��ă5��3��Ô�d=¼�zb����;����lF���=Л��,��(�)<��S�(o��b�2�ߵ�t�%�v�M��J�<�;�P%=�<����o
�������<�1 ����<������<L��=`�<>���� �t��� P���G��/���k�Jv�=��q�X�Z�xRJ���� �Y���
�`0<��?�ý���<��C������=`<���fh��䳽��׻V�+�?m���6<�3�<�{Y=���`Ц��J����< c�<LN��h��<�o���j���м����81����8�?=�jD��z��(b�<s,q=����鍽�.����0=0ȼ�y�����@<н|a׼�D{3����=�U�����< '<؈�8�ż̛����ɽc��pD�;&a��o��.蚽Л�<OAý�1�`�<P�׼��R�\V�=d�ڼ�K���=L������<�'�<�1=pQ�� A�;�qּ'��;�q=��;�)�V(^=�r���8S���#=����6� �ػ��a��b��
'� ,p9 �<:�J��,*=(�`= f��8`;� ���û���4n��b�����,9�i$�=�%}=�m��'��|=ȱY�t��8�n< 7:���<�ؼ�y�� ��)��E�=0�<`��;8t����߼ ��>��]���4�	�K�V���h����=�"��u��E=�$"<�B<�<gm������bM��ɽ��<��P;�<@� �������#����=(�<^�p�ͻ&=��Z�"	ҽ�1���u=^c	�����P?����x�����=#I�=@W��q��HYt�BB=LJ�����=B�=$D� ���Y�9 ���>()���m
� ̽���@C��������t�Ӽ@v>8,����J= ��=j%���;�a>=T�(b���8<��
��H!=���<e==�D��`u̼h��<�ٳ��aѼ@1.�~�C��Cv<��I=~) �=�B=�36=�ڮ���0=n��)d�,kd=��<���zp)���W����������l�<p$<Õ�����=D��=4�4��ʻlM�`�B���:�=O��=xh� >�� ~����=n`��`Y;��o������L=~
�$����K�Q����=¹~��K=.�=�s���R�<��%=�Ns;�.���w��T�5�H�y�g�І�=�������<Ȟ�<�u=�����D�@!.���=�|���hN�"��4�ؼ �ʼ��o�o񪽀���T�=t���{T�P:<�\���2��@�0��[.<`�7=Iʅ=8�6�=�Ϯ��+:�q��H��<xȼ�1� u�l=ba�=߈���x�=RdF��]=/�r=l�N�'�}=�6U;������<�c�<߁.=P�a�D�/l1=:����<T���D��<�9Խ,�� 1���ϼ$�	����<bY�h�μoo=u�Ͻ�G= ��;:�[=�!���üH�+��'����;�޼B
M��8�= Uм�\k=��<�CP=@��T��lO�<�=�Zc�sE=�������<�TC=�v�<�-=@ <�L�������<��� ���8"&��f��b�=JS=�d�s�ĽPfN�8�ۼz�0��Z�<�{��Mu=ȪA<J�/��)"���E�ER�=�+=
O�v��=��<�ɕ<7!Խ��N;zS����t��=Ԋ�<V��=d�����:�8.{=���=({.���<@T������=�s��r�=(+�<��v�X7�<Nc����i=�q��!:do��`��;P��;�e��C���A;� �=��=�p�l�<4R-��&#�P�PCi=؇ۼ�;����Ӽ=,=�, �qt�=�#�=�4%=��<(�:<L\��C>���< �{9N弽����pYY<��%�=`D����<>򮯽�%t���O=*(Z�i3=*��=����C�E��=.:U��)��P��; �m:vz罐u����#=�����h��
���}~ؽ�=�4=�e�|h3>�D���Ͻn=�o�`t�m�=X�1� ;�:�.ͽ5����R��R��`
�\q�<���<������=�'�= ��;��<�p��|��B����
>�̰= �,��h+��<���;�<@��Оڼ�Է�x�<��=l� �F���� 4����;����h�~<<�=�!��
�<@�1��:�9���.gp�^�=w���`��.C�=\�K��,�=�p�<��!>mE��|���t��J�=BsH���v��]�o掽��M=��6���� 7K;�=HcE�D2
�4M������p���༖�=_��=��;�<s��=��a>C=�怽��=\�@�E�9�5=>��=���=y�Ͻit�=�.��"n�= [=�}z��A>�0���{��qh<���<���<�⸼�`��.=��N���";@�\G�<��r���R�\����\�����?�<�z���ƭ�e{+=c妽t�
=�$�;'=�6�1q��H�(�� �;�v�@�¼4�.=Зݻ��=%�<ԯ=�#H<��w���2=���<� ��EC=�II��A�;�F=�s<j�	=�I�:�O���,<`��;�Q��8����<b�,���Nh�=�Q�=XM�Z�t�PeĻ���"�
���<��+���= �;��o������=^^	=\?��K�d=��< r�<����� ��l�� >��K�R=|��<j;�=�\��xVü@2�<T��=�0Ǻ��~=H�[��H��H=\"����:=�͒<<�ۼ�4�<�}C�3<=(�=x�S<(�b<�y<p�<@���`���p� ��a�=:=q#�h]K<�y޼櫼0�<��y=���~E_�0�^�B�f=p�m��k�=8��=���<��<�=<�$�=H��<��<�T���7����=諒-a>�B��4�7>��ĤW��z�;�y��j�=X��<J���!��GV�=�7��<{�x~
���������9�<�/=(^C��XL<73�f�@�(�=�W�<��;؍>����Ž؆�<�I����G��F�=2i��w{=���|�PE��t�Ҽ�
�� vT;`�<(�<�z�=Ao�=�f=�M;��H���޻�^H�9�+>��<(j2�X���P������;ڌ=�=���|���>�sp=�2�V�����~�r��q�V��د���C�=ܘ!� $o��8I���D���˽J�/�r�>�N�IC������NԷ����=�ī��Y>ya��`�&�L���?�!>�L˽W���fNq�'�ν��>p�Ի&���<���<Nx���]G�B(���25���A���={`&>���ԉb=ͱ`=V�H����=L9��E�=��R��&=/��=N�>5m_=r���W�<	��e��=`;(<�9����>����t�Hl���<~�=�.Y��C ����< �˻�Ơ�X{�h�<t:���;.<)���;@���8~=c<6�}Iv=����Z/-=���<� =m�����;%� ����͍;x�'<����u"=��;(�<sJ5=Ь�<�C�<p�����=xx7<8ȯ��g=� Ἰ�%<&z�=�<� �<(�q�8<�&�<�S �`ʌ; ��k��j���
�=§=S����>��݂�lG4�H)j�EHD==e<h�<P�D�����T<� J��$�=�M�< �`<�OH=Gb;=D��</Q���ᵻ��A<��p��<���<*S�=��|���'���K<3��=@��:��=�z��~e��?�<ް�7�m=@��:�V�:@=�żw�7=�=�=�=#=�%2<|k�<@<<p���ر����=�F�<���Xv��1��`P��Ιv=P��=�{���0�x�b�*�=��;���=��Z=���<�0��
i=�*=F3�= �I<T5�<���N���/=`
��->h-)����=�������d�׼UL���3[=��T(�QS���p>�����:��h,�[;Ͻ��׽�K}=�ȴ;�Ix�ȂA<�d��R��$�=09�LK�<�<t>�B�`ϕ�	���=H��@�pY�<ďM��N�= H�{
�ȯ~<��!��_ݽ�O����f�<���=n�~=xt�<�����(�  �`%�;�	>�0���K�
���+J�TaB�f��=2����1Ž�A4>`��<oͽ��r�7����s^��,���|��H#i��
=-�<T1F�4�����5���쳼5">��-��ț�AH�����=3��4>�τ�����	��>����2��޷	��/���(>�Γ<�N�:�=��&;��{� �"��%ֽ93���ẽ�N���M=�&>�h7�I��=�
b<h�5���>�~[��E�;��]���<�u�="e�=�!<����ڼ�㛽�7;< K����y<�qg><������������=@��<��B����Y�;Ǽzz��_h���2�FZ��K$;�`���*�����X= ��㐾��=b/a�ûy=_�9=Hw�<8fp�xQ��O8��0+� �<8��<p��b= �;	=Ƅ�=��<��<4���͉=x���]�9=ؐ����;���=d��<  �:Ȥ�إR<Xa�<��&� ���P���n�/�&�N�&��=k�=��ý�⇽4��No������.=d�<�x>=��ż�5������`*<�d�=h��<t4�<�G1=��=���<]Dٽ�F3;d��<��޼@>ں��t<���=0��,���,�<�[�=���;^��=p�ļ�c��Y��WU��`V=h�A��؝<� =4(�n�T=���=_L=d��<�V��(��<HEn<BC������7F�=�3�<�)�<a#��Lt�X�8��l�=�Qw=H߼j�b�,���ۋ=�{S<D�="��=�	�p_�����=&x=.�=�uH;X�<�����f�'�<x�=<���=�!�<7_0=�3���K��ż�낽�lA<CZ������l�z�=�D�"] �dd=�sB��Vg��xe�=�T�@�,� `���ue�F!��<ޖd��L9=��=D����������.����� ׻�&�a9=:���
���-=�Z������Լv�� &t9��]<�OL<4̍�&�o�
x���u_���<}�=�pW� �X��`��6��f2�l;�= t���a��<�>������l�$j�F{��K�6��B� �D�>��巼��=^����T���m������g�:J4�=�e��$���5��d�=���̼�3��/��=8��<�����ҽ��o=N�a�����H�[�2N�V�
>\"�<`lU<�|�<`�R<hs�����$Qż�BȽ����.������=�{"�+S�=:o ��޽��=�ǰ<��ƽao�$ּp��=�m�<T@；���8�|���E�O%��p񱻢��=ӓ�=`�T��낼h���5�;hC�0�[��Fg��PD���¼ dýlכ�Ԋ�<�@����@�<pf��Ă���|�=0�9��T��5�@=xz�^�%=8�j=|�< ���l��<�ݠ�'ۗ�?=�,�<���6�i=� �: �?;�΂=`�� » �R;�Z�<�B!�hn����c<�C=<|;��_�e=f-<=@�X� 6����;phe<bp�|��0�'�Z�q�4���
7=C��=+½&�K�@p�;X`ܼ �};���<���<(q<��;�}س��X� ꚺ�D�=��<8�R=`�<� ���@@;9a½���0E�<p�ܼ�����弉U�=����(����<L��<8�n<
T[=:��=閽$Cj��맽S�=��a;٪{= r�;�:	��D<J�>�L/=@��� -���=f*<d�����μ�8�= ����~�4)i� ��|�üx�>nYx=�Z���SC����<=�6<<�=ƭ[=�ͼp��/=�S�;�f�=�}���4�;d+��0an�6�"=�R�<%x=�-=1�<�g�,U%����ʎ�0Ή�5����2C�F�i���==��ƻ0�6�t�V�}j��4R�K�o=|���/��zȼȐX�����@@,;]z��df0=Ui=���@Nܼ(;����z����^z��V�i�=qa���{�A=�*��B��LoҼ�8_� �=�`0� (���#�a7���b���ټ�6O<�q=�Ck�(�C�،'��ch���%��;=�x��T��/�=0�6��c� ρ�vt@�!���L�����H� �6##���=ơ���?��@���������P�<� FV���X��v�+���ƽxq�=�Tj<@��r��`��<Jr��ϙ���Ѻ�iǼ�L�=ج�<`�v;X�[�$;�<p�;�'i����;7���ټX�%��.l��T=л��'c=�@H��謽�Ǖ=�/=����c۽p�$4A=�<��/6����:����zO���Ā��DJ�=h��<H�}<�S� �7�
>�0����;�������ԙ��<���`���	�)=X����2��'?=��B�0���V=*��O5�� �>� t<�'Y;���<0ݻ;𽅻q�C=\���νlV= ��9����%= ���� �;�=\��@%��_�<H_��?�;�	�:��`4=�� �}0=�
= ���ܠ�<�!��K��P�ϻx�����=���k�(� <�<<�RR=����z5�&T =���;�{׻X$3<���;Ȑ��@���z��8A���Ǽ��0=*:�y�=�B���Ǽ@����7���W��@ö�@��(L��fȇ�-��=:�&�ؚ<|�<�V뻰��;xw�<�|>��r�����=�Ľ�͓< ��;W�=t_���M��G򼫠�=�S�<\�#�.�; = d�9���ͼ���=t�ȼ@h,��Q��O<��ӼKy�=�Iq=�/��^�?����=�<���X�#= T=�X���ؼ����l����&�=�������n�̯�� �c�����yN<�����=���8��<���<n�� O=����<���(��=��y��웼 ̓9:~&=� ��~�a��	=�� ���LR=��/���N=D��<R�`���5<�ƕ<�`-� %�9�f�L㼘�U<�Ć<@�r<�g����=���L>(���3<�~<ȯp<g+��m=�p�=h<�]*��都T1�<geֽv��=�;�=�\��K�֥���}�= �b� I;�҉<p�v���r<?�̽��.���z��'���T=]#��14L=�=4�>�|.	=pWK<��=0���������5��\���O�=܃ ���!���X:ѹr=/H޽�,�`-6;l��<�Z'� rI9Tݼ�\��nom���`��� �z���F=>�x��� Y�;6�+���Խ�P2�NF=��=���<DC��� <�����/�<���=d󎼀*<����y=�=n`�"{�=eI��z�Z=)�=�ys��gQ=���Hz'<3:'=O�5=:�[=h�;�����_b=Ĉ����X<l"ƼǠ2=�� �ռ�]��<M���q���c>=�@K� �� �<CY���]C=��J<(vW=r������ �i�(fS���;pZ�~��rώ=hDq�@�v=���<�g=��e:�3��=bY*=�!�`�?=އ��<��<�e=��X;�]4= `�����:�K,�g�G=�)����f�@�M;�I{�*y�=�H=VZ�[�ŽPQּ���+Ҽ�[	=$벼�n=����rO�|�ͼ,��,��=Z3=�1"�'��=0Ǥ<�*�<}�����<Lϥ�,���9�=��<K��=x�'�\�I�?	Z=�|�=�D0�8/\<�-t<�i��r�=hfx�XG=6=�kK��=(i����f=��o����<�|U��<P��;0]a��#'�H}<�Ŗ=�$=`��;<��<���>|=���&��k=�N|�{х�������<����$�{=`��=G�:=��W<Pg�;)���S>^�<���;@GV����$<�u����=�L�[|$>\5��28�� �T9Bo����<>�8=��G�>�|�_��=O|��;x�����@���@$��vA�ȦS= �g��&μ4���Ƚ��=8��<ҷ&�AX>�[j�p�{�_�<ƦX���۽̚�<8���as=��Ͻ�н��� E�@B#� d<4=�< �����=��=�i:=���e�&���<���0;>{}\=�T.�`s��$x���}�:��y;4�F*:�A�<x|.<U#����6�BZ�l[:�0ǆ�������<ҭ�=�e何b =��ȩ�<V,X�/鍽l^�<O��G���p=�z��r�9=�V��r>cz�p�~^���Q�=�R`�~R�>_C��Q��x�z<��	��#��Y��=,v�Ш���:��=r�����7<�N��=(��=�J�T ����=.���܌=��A��w�=4#���ni=�n��8��=;�=��½�[�=�����a�=��=�����s<>��� ����h=O=���<�Sټ0�e�h��<H�����;:�U(�w=*%t�h'C�\Ս� jһ�����{=X�	�`�;<�<_O���= ��<
�=P�ʼ=ؼ�,�dyƼ�����J<ȣu�ga_=�V<;�w*=�r=��)=8�y<Z �^D@=lN�<�t=�<C�Ђ�<	�k=�ǈ:��=�A
��"�<���<���<�R>;L� � ��<�¼�3�=��~=j�f�����Q���N��^y���
=�|��-v<=@ܼ�,��^��@L8����=�v=8��=���<L��<_�ߜ<̕-� ��:J8=Ӡ=�D�=�r!�T{#��ې<ַ�= Ҽ�;=02�;���c�=lG��H�(=�F�<d葼r�#=��(��$K=���<h��<�c<@��;��<0鮻���7�;���=i�4= �E9��<�����1�H�#<L=4O��ʌ"����Ks=��K��l�=���=�`�<@�;��=��<]w�=�Ȣ<t��<@T漪����M�<hgʽ.�>��->(��<qm���6������!?=df�<��K�a���Zd>ހi�����K��m�����ǽ�Fo<	�D=t-���;�)E���&���>\�<@X����s>�!	��d����9;U���$�+�jT=�w����=6��j�	�*	�H#��0^�����D�<C�=.��=��= g=�,μ[�c�(�<~h�t�3>���� ��';�ں�]�轚�=��tS���>��;#�X?���H��Qg�k���;��T.�����=����@<�F���ʷ����2vt�q>na��Eg� '�9�3����=l����E>X�0[ݼ���>Y�/
����d����5�=���;@)2�ϧ=�;�ٜ��[�;P��O뽺[��x��+l�=�<&>^�&���j;T�=P�>�h�> -A�9^�=^i1���=��+<�\)>X ?=T��h�<Ѝ��S�=@ku�P���^��>�Eཬ,ü�K�:�=P
=`��h�z���;�=�|���h� ���<LO���7'<|���@�{<�軆	= �?<pk3�=��e���<w�<�o<��^�hSV�XA!��#�0&����< 7;�=<��<���<e�+=�<�a�<�ۊ�$\�=�T�<� t�m4+=蘺���<��=�Kp�$S�<htj�X�=� =l��<8��<��� U�:��Ǽ��]=��=g:����%��`�P�Q��慻�:=4G�<h��<���t��z�(����<	)�= G9<$�<Agk=?7'=C�<BA���s<��ǻ@(ݻ�ޕ<��<��T=�NU�,/�Hς�Y$�=`D�K�w=pt=�@f���<=��QG=`!�;@��;1�-=�C���*=Q݇=��U=���< �D<,�<�L�;X�ȼ�7i:b�=�
=���;`�0< _��������0=�w=x�N����:��E�U=\#�<�W=�Td= d)<����LA=ښ.=��f=�' <��<�*��T�T�<p�X�h|>6Z�����=d�ʼ
d��띇��V����=p�3��l��r�<�P� >���N�`���������R=�s�;|�d�8H|<m�.�§4���=,o�<�2�<�5X>���\�"�̈��K��&����<0U���=V@�Z^� 􀹠ѼH���n-��إ;FA=�Ss=��{=Zf=�n�I_=�2=��!<���=s"��1��=�T�����$�;_�=dxQ�N0��+#>P~5�����)�f #��iI��/��J _���Bs$=��R��@'�2����j�������;���9>QZ:��A���x�x���=��f���>��"<�裻s 
�B��=�ֽv<|�bf���ݽ�>
N=h�<M$u=l:��׎��ں;p�ٽ�ܽ1e��n�����=��*> �T�N�	=��N;R��>xd�P`x=�4�&�=�/�<���= r�92���r���Gֽ�0'=�S,�Dݽ��$j>�Q��\ȼ@����	=�L�<����:�4ݪ�X��jW��R1�03<K�����;D��h�<HC[��
0=��;V�W��L-=D�8�=�y(=pd�;d��#���a3�:/f��p��Z�< 6-��]=@~y<���;�f=x��<X�<�J���<�=�l6:p�5��&=��� �l<i��=�K;pY�;���n=0?�< �.��[T< Ha��X�f����P=�Y�=�Ӿ�f[F���X�n?�� ��?%3=�A�<��<�2��v�Hb��(�<޲= 8�;JK=k�;=*$=���<��t� �Y<���;<+Ѽ;���}1<��h=�# �|�����ʼG��=@w;��M~=�����:I�`CQ;��*�F�"=�޻PQ�<��=�H��x-=�x�=�̀=xr�<�$<���< \<\Z:� jm��G�=�<h�L�`�q��K.�^���1�=�c^=�8��J������Вg=d��<��l=��f=p�V��=8�|ln=�(=�mb= 'c�dx�<@tq;\�X����; C�:��=L����%=�E����b���`������A<�U�� ��@����=��"�՝����r�r!���/��$�=��#�(|)�P`<�Ƚ��&�R=��ѻ<=C^�=*�۽`Tۻ�̒�2�L��8�8�h�0����=�󤽾6����<(�~����A�� �ȼĺ�<��<�-<����<;�Nǽ��<��<.�e=H�U���L���콰	��
�T��=���D�]�=�!���5=�d������  �-L�̨ȼ\	��0����=�*��2����Tٻ]bŽ���R�=rc��"�eν�߮�d�������K=�S,=@\X;~W�1DU=o;��r��:<����(|>��D=p�=�P=�k�{�8����̼u���XH���!��`]Z<}8�=r�4�F==��}����8�=���N���齘O�<L�<���<F���_��P�K�������ؼp�9��V+=y��=���E���=-<��W�9м���R{���s��ެ�P�|�hn�<�?޻����0f�;�D���^��~F=��������q<,Є�tĻ<Bmc=���� �1: �k;pvl�yݟ� {6;8��<�$L�T6=Є&<�n6�;T= >� ^Q9@`���=@�ܺ��>;���;�F�;P�	�i�N=���<���;�}C����<$�<�8� h�;8�����a���q�̘�<�DZ=�ɽ�8������� ���lS�<h�k<�w�;�Ќ��Q�z�	��[-<،�=��;�3I= ��; �P���;�]� ����:�<BܼH3}��4���q=X= �HO4��徼��O=<��E=B޼�0�@��{��4��< 
�;B�u=��\<t��xJ<7��=�N=@���Ծ;���<�!<;��m��ﺼ��=@>�;���Xe��e׼��(�@$�=�M=4:��Ή� ����S=@ك<�9=;98=f�����J=q|<��t=�j�H*d< �;�>S�`�J<X��<�lN= 9���^�<����:�.;V��ʝ��;�#��h�)� �D�=^�=@ƺn\
��:��+��hJ�Ŭ�=�k����� [+;��j���ܽ��=��ǼޡN=˃=�������ј�v�'�c3佸#T�P�׻5zs=������@�+;���&����\��^��@��:xX��E�;�����p���r������<�q�<$h���<�a���|�T�~O:=ؙ^�j���O��=!ؼ��ͼ��&�����ɽ��� |��8��DS��n�S=b���>|��P��;��˽����d�d=�i�`�<9Ľ �2�~)(�í�DB�<n�= Y�Z��\(�<�~!��8�����,w�p��=�#=`��<�y<�"�:�<���m��A0;�^� �������#��=*���p<�<V�:��M~�8��=�@[;�;��>֙��E�:��*<����:����$t��?��V�/����<u�=�,= ົ�^��Q!�@�R��m�� iM��tD��./�pB��j򭽔Q\�C=�< z��n�<pA�ح�<L�<05�;�x��^��S�<8���=�됼���;�]%=p�߼�Iǽ�Ϸ< N79��);0$< �;�H9����<��!��v��X=��`�;,=�I���E=D:��<P�<�7�; (�<H�W<`�%�X�u<xI<��Hֻ`T�<�����<���p �(�<t�<�U���-< Q{;&�������I#�6�;SZ�xa�<�Lؼ��+=��P-��,�:�����&7�@�G<�a�*��fi���<N���)�;D��� ��9$���`�O;n�!��ּ�k��g�� �8H<D*�=܆��h�����"�=�ζ<H,��7<�U�<@2��p��xJü=�n=�W�0!|�������<����K�=7g =���|[ż�����~�;8%<��� }�:·�*��P�#���wC)=䄄�Xw<*
dtype0*'
_output_shapes
:�
�
Const_3Const*��
value��B���"��T���`����<,eʽ̚&�(���_�=ą3�0k�=HDZ=���?e����l���ٳ����;�ā�P*	<M@�=��/=�$���`;��,=��J�T����?<<$߼�3'=�ߒ��,O�Da��|]k=�T���X���<��<l�ͼ¶�=��V=^4�=�T�HT<Z]���M&<$0�=�?�<hg�� =���<82��T�� �N;�Sd=�˽�xH�=*QN=�?o��H�,��O'�=7���Z�<��=�0=X�<e�
ȍ�+r<J�J�P��;!ҹ�A�<��=��ȼ4 =v:R=�X.�����vJ=ЛS�8W��"׽��=l��<R���i�Оv=�^̽rւ���"�@g^<h�p�`>�<H��=\:��d����l�.%5����$r��K����r�=�~=r7�xϼ]͈=8���P�=������=�&�<m��P��=�&���=@��;��a;r>2�vw9=��=�b�Ԣ�=ҏ�B >���>(y�<�M=P����̻4U0�0�,<�ue=������h�n<���M�&=�5�VY�@v�;xkM<x�q��r�<�]��b��4
ļ-+� �<�$Ļ �H<�y��Օ����r=��F��%:=��»���0<t�0�\��~����0��*�;�:���(<�u�; Pd�d����Z�<�1=@�B<X�A<�Xf�`����m������5���::П(<��ڻ �;���<�𼸪Q= ����ŻLM��D����,��pR�< �:���؏`<��_<����"K�X-㼌�I���８��<}�'=@��� �ڻp[L<h5<d��<�X���9ͻP�<���;���<��м�5	��BԻvG/��5�������)�(�_��	C�B+-� ��; �мȢ���$a�ZՖ���9���p���GD=�"�<�ර������Ӽ(�I<�=����m�=���L�˼pu�;²4�������� ��;�^r���*���3�pd�D��8�D��<���<��y� ��Ks<qz=������7�:���hS�=`�߻��=^(�=y^<�Z��8���3��|���`�;
v�X�<�X=l��<JK����t��+�=��!��r�� 4��. =���<��:<�G�@�<�-=@���(��x��@VK��7��>~=l�=�Ų=�i|;���<4d��(!�<g�}=��=�I�:�.=6yx=A�x=�=��=�3�=����ڳ=	(r=���< �8�^�;\T�= P� 'y:�d\=���<�_|=׻νL��<�<$ۻ��S<��m� �&=C�< 5��(Td=�b�<`�<2�w��E=��*�����`��j$=xM�< l�;�	�<,�=����7;У�����<Ys�� �9=�=����	t��{[�䷸� �9��:�3��8����a=��������m=(�P<��B��O�;N��=�N=8�¼]�=����6=��==��<(׼ϔ�=8��<P��<KD�=��`'c;���=@�;�ê=إ�p8�<�ш�����P~ƻ�n��L3���:�<���5�>=�P���+���(�����L5����X;�	�A,� �=�;�ܼx�R���3༤a�< k�8�<P ܻ@��<���<5�`�z��·��,��gD<�kC�`/��8v5���<���E��b]�`(����;\Ƽ(������; @��pl��'Z�w���S�ȗZ���<@�L��"���0U=X�ɼ�c5=����Z=(U�<fI���ɻ0V�;(�&�:< 
\;'�#=�^�pA� �`�D�`�NǼpC����:=�P_�l�д,<@ٳ�9�<W��]�<|'�<��	<�L�;V���׾� ��j�9���.��>��3Ǽ �k����:�	ټH)���qȻ@��� �.7�ぽ������� 6�9��<`��<�ؼPd'���p�;�j=@*t<H��=����Ӣ;p؁;�uż�I��Rü ���X�ݼ��  �� ��90ñ��,�<��<���IW=dI�<��F= f���
�;H��H���6=}�A=��=H��<�؟����=\���T9��T����m�<�%}<(0�$�K���S=8-
�!�$=p$���=�~���=8����m	=8p�<8f=�n{��>�0ŀ��� �೭<���=�F�=�Æ=(!=f�D��<��)<�c= n�;���<���<��
>Q��=C=jR�=r	=�=Q'#=h")< Q<0�=�su=�x]��o��F<Q�<RI�=K�Ͻ"����:A< <|<�ŏ<�2}���==�¿� �E�= ��N=�哼�$=P��<��<��K�Hi�� 뼈�<�"=��.=� �Cs^=ԁz� b?�8���I�=��&;�>�<@��;�h=H��<��l= ��<4�d����;�g1=l��� ������<��=`7'���;ʟ�=��`����<��=`Y<H;�<�r�=�:<� ���e=b��A�6=��t=��h��<�����;����=�x%��L=�y�<\~������`&��R����=е���3=pc���ۼ����Ϫ��)�X"q�d| ����:@��=��<�%�^����|� �:~�o=��ڻ@��������<���:t���y9��c�������=$�C�lS�@ri�l��<��*�V�'�Ѓ�����h�缚�"���Y�<�Wh��X�;���@�z�nP$�����Ђ�;��D���{��4�=��<:V@=��d�fz=�=V3(���<@���P^�;4��<��˻��A=�����< �f�����vĻ�t��V�K=��м��˼ ���}��\��<�m��s�.=@�;@Ԩ���)<�u*�`�����;�ۼ�8K�D��<𲢼�RF�5<X�b��0�@��;T⥼|9=���@�:x�Q�|��<�����.�<`�ʼ e�:��� ��:�Bt=�:�<(O=����<0�	<��L��+|<PA��.����<@_<��<p��;`k;�x�%�=��=,���$�f=����h<�V�8c0���ٽ��ǽ��o= *r;|P�=ৱ;(<�Y����==�=�][���< �<����I��(6\<��F���޺�<4�^ �=(�g�oܨ=꺓��=@n亀�úȣ�<���=�N��H�N<0��;8�T�p&H���4=={2=���=���< �<;t|�<A޼��<HN<@~�ɶ�g��=M��=)�=H�#=��=𹷼��C<��H�J�o�H=�6=���@c!����8C����X=9~��,lo�Б�;P]�<T��< D�:ʫ =��}�h'<�r&�`�����:<�{q;���<��
=��e=�ڟ�f�n��s��%<�R�<X����;춈=Z��v��6A�19]=����xh�<�=}jt=��G=/&�=s�:=�0�H�k���!=������<8W
<��2=(h�@���t��<����U=7�=@徺@b3<��=����\f���z:�3���{H=@���<��V;����e��r&<(E�Q�=�{�<L�����ּ�W�<����^�<�l.��L�< �~��3�Ԟ�d�Ǽ����xiL��&��=Z�=@��:��8�����0������;RS�=��E;P�O<�c���a껀�y< F�@�ջ�ч�@�(�t��<�W�����^�:�#=�é��d༈�6���D������;��c˼�X&=��:�l<pc��@&��Ӷ��dT����; �:��`�	�=��<��X=����tԸ<�O;=K��\=�ϻ`�y;���<����"=<����=��C���D��JI���k�b�Y=�t����еM�H2(�b�;��&�e=� ����:��#<��h<@�;��;����6L���=�j����;ؕK< ;������a<���}=��&; �d�PC�<4�/=@��xv�<0���� �;�@�h�U<��3= >=РL<0��G=\d�<�>��=@�<.j �Ք=���<\�0=�)�p��|��b'?=�Z�<�ݴ��Z�<�cļ�ؼ�O�(�i� �V�n��4�< �;��= r(����;�8��0��<���=˩���S=��<@�2<>hH��� =2l��J� �T�$��=�����=ܠ�5Y<8�;�_ێ��1X=F3n= �����6=�Ee=�C �����P	Ҽo�F=<��=������<�|�<�� ��������<`Ѽ���0+�;�=��=���q��=�H� ߀��Н�@,�ιC=\�(=nq�ܸ�����#_�|�<�� ��x��k[<j4=`=t�����л��}���;�����A<�"�0�Q<�\�<�(�G�< �7��B'�����	; &��HC˽�tj<|h�<]$�����N���P�<����`�<���;�9׺lX�<f0@=��s<< ���D�,�<����	=��=��=�U�@�m<4+��M��L�j=���*�� d�;Rx�= 3�_ڊ�O��&�y�� w=x�ۼe(5=Y��@�m� �T�\�L�x�	� �9�}l�Ȗ���N*�t��<�����"]< Q�����<@�6��� ;��j�,2ݼ<���d������=r70=d�ϼ��9�J��`
$��=��Ai'=�Qb<��=���r�� ��<04� ��:nԼ�d�p���L��0����1�Y=�nļ`<��pË��랻�?H���>���+� ��<`tx�0�j<(ݼ�+�� ���!p���; է;E�<�I=dT�<��i=�������;"&=Ec����< ;������P"�P�Y�h'�<T��&o<`��x�#�T�#�h4x�m�x=`���ʻ>��V��`�h<���:,gz��;=8{��h![<H� *R9 �:��軆�
���I�<�vL��;@��;�+!<"v��@��:\���0= �: ��tE=%+=̊����<�+�;�,����)����<���<��.=p��; -v;�
=��<l�ּd�<\S�<Y� K�`s�<x��<�L"�~��s7���b=،.��]��ѐ��
�#G��L0��࣡�hhݼ��ݽ�����5:���<l�ݼ��ռFN'�d���0�H=����h�= j�����<��ļTN�< �8�X���h0=�nF<�ֻ�xK=�0A�|:��*N#����8�3=\z
=$��.@==��=�r
���x��r���'=��L=v��Pq<_C<J���zP��j_<~�VV��
u����<��Q=�ML����<�򏽸�0_黀pU���=�]=�Ľ�Ƽ�4�ĝ���`��:�< ���,;}l=E<b�*�\r��x��p邻t=g��ݨ<<O�@-�:���<�k4�d�;�����ܼ`�������^�������8�輬ͼ'�Ƚ�V�;P�/<Fn��ܽ�^@��8��&�����>��P&D�\.���?�<��.����<5B=�;x V���y<�)����6���T=�E�����x{]�jQy= �z�k7��!5ŽvW4�Wde=�����<*@)�HY㼰�;n�Ľ $r:Ƶ^�X����5���6��,�<�G!�8�r�� T�9��ݼh�<��/<�]� ���{���^9���6<l��<V�L�CΜ��d�8����}���_�8<D/e=�w��d��<�����,<!��m�4�f����t���5���=^�_��o���ڝ;PP}���?<陏����� A)�x�>�8�<�K��-�0�<�¸��@��:'�"=<�輀[S;�7=�Շ��ڋ�=M7����<�2���d�8�_����p�=<��ʼ��@�`����1�vr���Z��]=P��;{C��d��`t=0}��%ٽ��3<*$��=.���P;��p��(���2{�
�8��qX���6���[��s���#�<��ν�s������|{�<Xpz�,N����<�q�<���x%�<������ļ탨��= �{:iqa=��4��<�A�<��z<r�O��< +;�4D�2��̣ͼP��.<x�\©��ᖽ�5=�,I�J2S��E��P�M< ���,ڽ����@�����=�,���>�= m=���}6��ri�(��F�;�~7������=��=hh�<��@o���W=�.���m<�\�:�W�;���<�5<Rvl�޳~���=�|���S=lڏ=��R<�8Z=��F����=d�M��D�<������<��=�����+�T8�<��׼����p��<�U=�&�<����_�3=�]=P�
��˦;���N�>Iڽ)�f=4��<�đ�D��<Cꬽ<����8=�WQ��̹<�0s��4���1<�W��I�=\q�=d]d�I����B=��f��8����Zđ=��|<�<�Ƥ+��X#��YŽ0A��@N�����;�Z���%=��=��ż��½�3��^�I���(���r�RMǽu.=���~n<�8�=���� <N;Px<d�=�׏;�	���r=�Z�x�a=(@�<�^�:�.a�0��<�*�<z�;�Vy=��<����1�=�dW<�:�;�Y��h��<htǼ,��<�r/=`D���1����<�\����=4似N������ ��; ��7p��84&��V��z��F�`Sg<tJ��,r�<케+6;����([<`5�s1=�^�h#O��Uջ@��:I���< _Լ�M< @Ը��;PՓ;�!���y��P��<hU"<@�!<�) ;��;@Pa;��лR�N�H�0�`���p�;�|�;�<�<���c=Ȑ��P�4=0Z��~Ի�*��T���I�@}R<E�<��;���<��@<Ă�����`�u&� 5�̏�<^�0=P�� 3<@�ź��<5= 2�<ے<���;�V���<�-u<��̼`�ڻ�����ȼP�����Y��RK; d���%�hԝ<`�M�H<주�y����HW��3/��~�<(Ck< 0�:��غ����͆;_�=0~S�#�[=pj��p �� 0�9b�G�H��� �z���b<�I������ j�;�D[;�k^;0v��6=�=�!����;xI�<X��<�벽���l�b��L�=�����=�~n=��i:��0�4� �K;j/<�"fR��ȣ�Y7=�d�=�?޻d����&��d�=�@w��!=�̫���$=H'�<��<�L>�@َ�O�=���X�g�l��<�
=���<��L=P�<�n�=Й��m�-=
C��䕻<h�i=�s ��@��6c=���;���<��=7l�=�X=r� �k�T=MN=Ht<
�=h�<�>6�����<�=��<�p�=����
�?�&=�ޣ<8p�<��F���2< ����Q<<�=G=��m=����yω���^=h�������wĽ�p= ,<��<�.u;��<����8���f��L�<���z�_=ڝh=�����4�8���,)����=��g<���j{��{=TӼX򊼪>�=tɀ<����`F=��|=��<h�<s?�=\=�UD=��t=�+�@na�P%=��J<�=�(�=@��< �g��l= ��:"J=�T�v�r= ��< ���8w��ao��%'���9=�p����<8��
�@�F3J��� ��J�*�L5�PxI��>= �������L/�lV�<�A_<Ɂ =���6�`tZ��=�G���iR�\�H_%�T�ټjR=@x�������Bǻ���;`��Z�/��3Ǽ��;��������:�#�L��<�m�:� ��;-� L2���6�`DC����;�a&�R�"�!��=�r˼��<��5���=�<�./��4� �S9)�=�|�<���;��=Է���<`��<�o<��<@�8���=v�C���ݺ�l�:����l�	=�_":zo=(N}<��ȼ�s<��u<�]<��
�;�h��QT���<P_���3;J:<�.���<�
M;�!<��.=Hmp� Q��ؑ�ȗf<�\<@��:�� "<`��;8X��C�j=��K�WQ=h�8�<�ݻjG���b:85���T�0��<��L< �<�9�<�n=�M�G=;�H= �p�Ň=��1;��;أ��`���Zڽ���@��<��1=��= ���3¼D��`�<�쀼�R������^M=��<�Nx���J���U�=`8鼗��=���I�y=�0;܏�<�h�LY�<��<=�!=V��hR�<%�X�Y<l��<B!=��=�[=5�=�dl�0Nd<l'�<`�0� �y����< A�9F�=4�=���=K1=u�+=���<ty=�	<	�<P="�= �p��5����;Ġ�<zE�=���P�3���=�}=@ԭ<��%����<Ƙ���#=��< ��`C�;i#����]=���<x>H<����n��p�鼂#=h<T<��<:,�@��:��W�ܟ<���"�=�y�;�G��/V=&�=�=<~A�=��4=��w��2��&=��?� �`� K�<� =���� 0�<ߨ�=D�<�d=�=�3$�hT1<�h�= �ܻ0|�;('�<��q�6=(JK=H\= �Q�P�w� q�:��b= aD��Q�=�&�=�0v���A�Nꏽ�0��O�n=T�ļ��<���:�hc�7#��T���x�kP���r�<T�<dV�=�0û`4;��~�v>=m�_=�=�n�㌎� �;q�=����`�<Rkq� tƻ������=Xzc�x��������W� ��;D��vf��k;$f���7�B�r��1=�BG�+�l����<���8V�`v�;PH���9���X>0w��<�}����2=(�U<�4� �u�������y=�M8=0o�����<xt����=>XY=��Z�!'=ؿӼ� =Ͻ��\�;@mѻ��.�Ц�<H�<0X�=���<�t���<�?=�+�;( x<���;I��@=p/߻(x<���<L��,�<p֨<��<��= ���=ȣA�Ac=�+g�8����(�(°<�^<�����p=<Ϫ����< W��)�W=��L��������<����C�����=�I==��<.A=��= (���Y=��=�}j��7^=�'׼����3�te��l[�P����3K= &�9(=�=8Q�^g�Q/��l�<�O<����i,<�+�< �;�5���ʼ�&��z�< 	�zֱ=,����̤=*-�@������ ���[<;��=�a߼"%=��w�@胻 �W;�c�<|�=���=���=��S�p �� c�;x�V��0s�PR3�d��T5�=��=���=�<o��=����%F<��P�8�� �=��=�Ȼ�˧��g��@G�;:��=�?����m��N�<��(=�=^'����<1ܗ�(�.=�sv�`��0�;e���M9N= �F<��7=�W�I����O:���=`�u��ܼ`i��0(><4]��h��h�)�:�H=��4� ���BȌ=�*=�=9�=��q=��x�C��$N=�j�� T ;�` <G	=�0� �:�K=�?��*�=jC=��뼌^�.~�=x弰�P���r�魙��V#=��D�̉=h��ȓ����<��<8�����=o�=���ȚS�>Ǧ������Re=&���;ֻ��b���a�m콬���56��}��◐��bu=TÔ=XX���}�<a$���$=e}�=H��=0ð��姽�W�;�<H�]��<=)�����;�N<��1�= �#��/׼`����ڼ��-;A؋��\Q� D�;�D���1�֩T�4�b=�{>��R�����`��<?���P�@K��A�喦�?�%>0�X< ]
��l��K�=���;�G�pF�;XD�Om�=�!i=��ۼ���;pY�祯="�=0jW�i�P=�ؼ�s�<3�ɽ0�)<l���gc� �g<�]�<�o�=���<�8��D �<
Ռ=tY�<�ڠ<�s�;���mb+=���x��<P�&< ���<h�<�ց<;�=t��=�E���!=\�0��EH=���(�"������<��#<�oM�I�&= �� '��`��;֓z=�%������<�C<ȷּ�r
>���=T4�<l�<qс=��}�=��_=�ʒ��h
=�0-� Ţ�� üh��,�i�+��(=�<�=�q���M�Ǐ��xv<�>�<4���v�< �`:�A<�{'��#��Ƙ �p�ɼ �;��P=ĸ#��~�=Rɒ�4��N7'���ý�l�<V�~=�5��2<]=���<5��B�Pr�d�a=Q�=hE=0WL� ��� }S��Z��`����.��F��P�1<�=�K�=`����ƴ=P�&�p6���컸e'�giz=�<�;\�������F<�k��y�<^�0��`_��'=nЊ=��=Z'K��<�����<8����'��w��l�E� =� ��"=�߼��|�0���t<��мX�g� �ι !лx�}�R�2���� �L<N-Y���s<@$�<p̄�x.�<,=N=���;�c���]=����l��<�9�<�c<ި��pԮ;�::<g+��<��=x�����/���<�y0�=t�s��,�����#젽�=X�r�
=�8N�Ƴ��	=�tT�`�ڻ��=`d0=�c缔�=����e��E�=�O�a���2���?�н4�=�`fe�E-�����j\�=lZS=�C?��Y<���5�<�x =f�=p��J*v����>���<�\�<q*��࣋��4�m��=���:tk�������K� a`�Bh���9����:�=���5.��^���U=�k��xr7�4���]X;66N��9����\9���ek����=���<�Dw: ��C�<���;�\��;�;�E�G>=�\=N�-��?���s����=@ =��)� n�<��鼀ͼ<�e������ ���� 1����ջV&�= f��^I����b<e<I=���< ��;8����ڗ���<���1|< =��㻐����;�(�<�J�=����U/<�j���$T=�!&�l&��f޼�h<�湼�)���<X�ż@�D�P*�<|�i=������Q�<�c�<fG�W�=*�=h�~< ����h�<�	r��y�=xE�<�Y��'�<TiC�b9�`���D��HV.�b���Ϛ<�F<�}�<ģ���S��(��`Cb�PF�;4��p�<X���aP<�߻p��;�:μ�1���=Pw��S����=����y���q�N�*�$��<��="��0=�^o=�?*��B ��z�D�<A��=pD(<��k�Ht����޷G� �&��D����������z=��=��S���i=������𯬻8"ӼfnQ=$k�������}?����<��C��p����9<�n�b�=�]�=��d<V�f���������x��Tj1�@v;��ּ�/�h�z<�G��Xu�<��4�j�7��"q<hܰ��?]����0dӻܳ!��x;�ᠧ��I:��Z���t�P��<.�H��@r� Q��ؐ��xI/�$w�<H!i��w$=�y ��w~<؟�<8$T��Ƚ ~�;­�A}����`=꥽P!W�b;z�n��=�����Q���榽!,��d��<6�����<����0��Ru=��߽`,[;t��<`�p��\0� �)���^���� u�;�^�$̊����ԏ��j}q��FE�8����et��hV�GN=��=M��ƹ ����ػ����R����< ��8� �����)R��Oj;H"k��D:�(мH4� �:@�U��zw���7����<Zp�vK�|V����I�$!�+���,f����<T���p����@����dڼp&��w��d¼�1��P�g=��H<�ZٺW����<h<��0J�;�������:�[����u�a�DG���=t�ּ��.�@
�d@*��ն<R"v��ۂ���3���<��"˼XFv�� :=`�μ�=�Y����&<H�c�������j�����$�z�F�����દ� ;L�g�Ժ���V��[�=��(:�����/���=�p9�`����Ҽ�� ��π�x��@yĺ��;L0���<�O=��;Z�u�P<*<lJ�<��=�� =Hi=������-�DG�\P%��u=��|�W]�$70� �㻒7��j��ztʽ�Q
=��=�����>f�=X�:��_c��y����<<՜�Ư����<���=a��=D �<ꍽ���8\��'���#�<X�<H$�<P!?�j'=�Qj���:���>@t_�<��<\�=B��=�Q=e�-=ڲ����=�n�� -�;��h��G<B��=�9{��L��V8=�����x��6�=r�=H��Cs߽��K�C=a=`�h��>W�@И�]�><r���5=$9<�Z��oWr=�p������0 =@.;vO�=�)<�0���x�<�X�<��<b��=�l��'�����H<�����,b�Q���:L�=0=X�k<����P�\��� ������e <|ٲ�ܚ��`n<�;=�>�Ĭ��m��_Բ��1�;(u|�������P\=v�.��,=�v�=��_�@���u�=��<\��<BZ�@a��ތ�KFt=���<�l�pkм�켼�^�<��ɻ��2=�2�<�_x�dڢ=����	n�xDg<N=X ����<�v-=������Y��_='� � <|.ϼ�$㼍Y���*���X�;ƕ�|)��(*:�a!��J���R=�D����g=h�	��=�E� 6N��a��TG�<@�ݺX�.�b~ �,r	=P��teL=P��;�n�<@�/��;:�P�X< ;^�X���K=��-��Ud<�<؍=�LJ<@�g��m��ջl���1�0�H�|�փ7� :�=ȝ�<�P�;`���.� �Ѽ�lV���7<��<h�u=���<��ټH��R&?��9=��#=x�S�H��<Tv�<��1=1�½���4v�� ��;`qI=(�<<Ƥ=��J�4y)����=��=(��Hѽ<0�������Z�<�q��bp= �D<^W_�b�=��3��D>=��X<p��xk���;�0�4<P9�<�p"5��
"=@g:;pn�L��<(=����< �q��[�<́��O�����*��߿<Ȍ�AC=\T�=���<�m�<��<����$��=��=qT�0�� M���Ǽ�F����&��s �R��=h�ֽ���=�g�=�$�H�����<�&�;�ݼ+k�ЌZ<R˰=�ݑ=@���Ja����o���<�<d�,�B=0'��]<ܕ�L�=��W�>�=��=�f>���=��< �<%��=��
=��U�^�^=\ q��,�<�L� 帹�7=��H<��Zh=��;�f$����=z�U= 1��ݑ��|��<���=  +���L< w�<Mn>�T�� ��;XU�<�ַ�#��=�����Q'���<@k;��U=�qC�`�W��e<M/M=� �<�ҏ=�5Q��b�4��<�ɗ����᤽�"=t��<R0\=-<���:U�Ƚ��9���2����<�~��xv< '�<�d��Da��n�j�*����=t)<�c�2b��D2�<��*����n��= ��:P&E<���=R�=�b!=��4�&�<�/��⟀=�n�<�E���=X/<j-= ��9>݆=�0=��6���A=�׀��&==(qL�O`=02}<�4�;0��;c��%���Æ=���x�
<U���%�s9ǽ� Z���\��'N���O��-<�A�<G��C=a��,�\=���<�w=]�Tc����?��V�<�q�:��p���F�X9�<�{��V`�=��<� ;��1;Ѓ���{�<��Լ�Q���*=�Y�ϻ�M]���2=�\�;����2�]�p��;��X�@��Lr��T��K<����>���<����I�����<xZ<��?h��Xm<�<��=rN%=.���f���1��ӫ=8-i=�j5�,��<�@<��=����@e��8��P��`�8=x�z<�ج= ����j)��3Z=�-{=��5;�!7=���A~�� Z�<�+ɼ�i={&<�'��$!=�Jۼ�
$=�C�= �@<�3��,f�<�x5<��%�̦!���Q=���;8�9����<��?� �)=X�O<�hW=����၇��9���f=�H��yK�=�s=Lb�<��=��p=�	W�b"�=Ӌ/=�kӻ0#��$�$���ڻ��M�4�<�N��H=� A�x5ɼ�S=x�� 4�:(��?1<\5¼�;I�X��<ND=���<WY�����@6;��=�N�|��=�ΰ��d��pΧ<�<:|��~�=Ï;��O��L�<P-�<,�缪�_=�n<����.�<���^��=�>����w�<�Q��
��vNY=��;�<\j�=ʁ=h�D�pQ!���Z=��=t������:=��=ȩ��lk7��ƻh�H=��=����HJ����7��6м� �<�I��Ā�8a��Ę�=p��;�`<�8�#2��ȍ�<l��<p1��(���{V�J��P�k=P�;�`Q=�>?��/��M��v_=_e��@0�;�덼��^*�=h	�쏘���o=+�0=�=�~셽�W��oM���U����</�2=`i=lj�<�V=��;��[�Z3m=��e��:�<@ô�����h�=t��< ����\u�<>7=��ػ'�� p�:/(�=�%0�mU�=�.= r(��fj�>�ǽ���@��=N��`����^�6G������G��@���oy��1�0=�~*=��y�R�=�2�����=�l=6=�=���烽`` ����<��;y�<�놽\j=_�^�=(%�<�֙�ж�;,,ü��<�#?���u�X�=4�����ܻ�뷻D;{=`Su;X��: ���<�b������-��r�'��ǽ>><>RjF=�#9�y��h�=\���Z,���<0�b<!��=أi=��T�P_μh׼f'�=��=<�3Ef=��;L��<x����;`5c��,�x��<
=��=xWi<������a=��=��<w�g=�d.���ý%�<������&=�<`P˼ea=IC��s=V+�=��I��v*=�/�w�= ��;��}�th]��t=0t<�������<�抽��=�8=`o�=�Ӽw����{/=�K���C>��=��<�8=ʓ�=`�;���=�c=�I���<�%g�@0��淼� �<��<��E�� l��	��xB=�ϼ<s��H[�`U�;.�M<��8�2<䜢<Џ�;k���pv� ~z�<�s��= 7Ž�	W� ��:6|�
��EM=�_�l��<H�L�$+�<�1����<���HB< ���0(���T�=з����~�P�F<*��$��� �l<�ꗼО�<Lޟ=V@=���]`=$=@���xdڼ�p��Hi=fZ��B<V����
�tԆ=*��=�Fi�D�%�@����ּ��<��_��c��6�`� ^�=LW��d���ϼ�Hս$?�<�9�<P4�������ň�$)��P��<�~"��=L���T0����J�=+D����0˼�ba�r/�=��;�v<�GA=-HR=��Ҽ,�Q�T1<meʽ��V����;t��<8eZ<|��߹^=�!���?�O�=�\ּ&�&���h��o$��7a=�h<��3��t���Q�\H�<z�$�]G�l�&=s[S=Xb�T'�=�G= �H�dkݼ����t�=*�i�H��|��6�O��j'��� ��t�:A葽V?���)�=���<�����U�=�Q��9��=�ݘ=h��=���r���3:���<��<��4=���{U=8O9����=�=@��:��i:�J�m�<�W}�Wh��8z�<�|ƽ0�c�`m�:�=����D�(�z�� ��<A*ŽNA�����j��i�s�L>j�j=�������<�
����]z< �<_X�=b�~=����
 1��zs���=���= ��b*�= V��F�;��,��qB<pQ�4~\��(<99%=3->��g<p��z*�=f�=l��<x~a=�#5�MԽl��<$Pټ�H.= �O9`����j=��¼ę�=��	>@g;�?=�X��?7=�!�B�����c���p=��H< U�����:�ǯ��]0<8��=�˛=�����}�x1���J=�Ƽ��8>�=�>�<
*=�
�= o-�?�>!�S=l͵��o�<l@_� �;p6�;X�:� l���揽`vG<$�O���=豳�%	��Ey�� m0:B�<�_������H�<<��;��׼,H��Bֻ�v��0�;��x<<����w<Բ��ժ��Y)�*�,��Y�<�sʼ4ť<�2��0��@I;@�[�`^\;��=3�=�3ý±�� �o<�:0�0ƹ��U�,�0��/����-=�d�=D�	���=@!�|!������2���e=/���0(ӻ ����3�<v�=�G5=�)�(m�D��<PH�;�t
=
Sm�_��Ub��
0�=�@�8�μ��F�՞佘��<���/=<�������z<�2F��k�$��<ht;��*@�I���]?��ֵ�>r7���M��%�<�Y=��� ��; �9�@=��<T0��Z,=�Ÿ� .;@��:����(+��.�@�'=�}��҂;��y<�u:�%�����<蛽`��<Bu������)��뜽��<w������=,� ��}��n�=�O�<t����#�������N1="#n�l�N����0���b��`$��m<�x�G���Ŝ=,}<!9߽!��=�\��o�O=�%j=3ߚ=ȏ���X�`� ��d����=�=���/�4=� �Bc�=fb=D����F���� � G�9��J��͂��]F<��u�*t� �Ǻ�)�=�j0��좼��4����;�_���j��1%�i���j��"�>h�d=��h�2����'=���� �@~K<�K�;\��=�=;.ý��\�����=�C�=���ȱ�<?���h�4��0�ռxzؼ��PV����:���=p������Pqa=s17=�U�<��=����@�ǻV�>��E�<p��%����<`*��X=�o> 2<H.3<����	�K=<�������t5Q��B1=l)��J�� ����I��&����=�֗=��A��m��'���%/=�_��>	!�= ;���A2<�~.=������=|h�<�V���;��y�@��;��r<h��֖=�^̽�)�<`���p?�<D��Uh������(�"��]�W���m*�0ϐ;�a;0��;H�0�H�`�H���H��<L?��<�n�<��_�}ؽj�\��y߽Xc�gI<Xζ� ���@a�;�خ�������X�\��pm= o=T^�~����R<�2��䮼cԈ��uY���`� O�;a�=b�Uc�=��!�8��m��Rǽ+_=/Qý\�	�6ϓ�R�S=@�<@��:�`}��j��E�<f�<�C�<˄����������Wc=YH�<�Ҽ����ڽ�:!;x����< Р�닽��=����`�;8"\�T c����P�2Cg�/ނ��茻7�:=(fA�a�`�>�W �@�<<x
#=T|�#b/=����x1��į��iD�i���}��(�T<f� �{���`-�<����l)�<vjĽ𧑻ZqW��A��R�\�ҽpW����ݽj�V��=�1����{<�2�<P���Z�,�p�3�>����J���Y<�qh�����0� ���c�ƽ�#6��5�;�\�Xp�D��= �(<��Ы�;�A���L�:T�<�L,=�S�b���+������=�K<Vq����<�a"�=@<$&�<d�[��5/� ����x˼j��@*�'�����m���-��g6= _"� ����_�x"�$��� �r�8�2�f��v3�Q�=�� =��O�ޢ�B�=p��vI���x<���؊�<���A׽ �߼zZ�E�=���:�ռ ��\��ٮ:wA�!�����V�(�Լ�$Ｄ�J����=H�Ԇe����<`F�<�� 1�:CH��<� Z��1ד�/<��>�q<�<��iM�@���rT�=���<����D�H�4=��"�(��|�9����<�	H�(@�������@���޼B2�=+p�= C�:Џ���ļkS=�w!�v��=��=0�Ƽ���r� ��^�=0�!��+8�hР�XL#�2����u�T�j��l�<�.�=������=R��=��ƼX鄼1�= ��`eĻf�2� �;*F�= �=f#=Ǆ� ����M���ϼ �j:@���`~^��d�>=��<����<���=�`���e=�3û��O��=�,�<<,��p�V<r�콒��$��<��`�N=@��:;�����=���<���k=@%�<p�W�'�Н�;�j�=�%� ������P�>�(ٽD�<��?�T�ŽsO�=�a �`�%��*�H���=:jR���;0pH=p(=@`:;&�=Llf�eˑ�z*��8t��!�b�i�9��=��<�L,=�4�<��h�t���(_� ��;�:f�����-�0����R���UD��&���۽ �	� E��,���Ͻa=Rjj�HV�'= 64��e�<��=�kK<��4={��� ⼂-��d�<= p��&��`��;�m».J�=�1#���k=p&�;�"��Gw=8��pF�<|�<h)<��:���<�.=���N�_��~k=z����:�ژ�������Q������؆(��|J<�c���I��vW=�Iҽ��S=pu���%I=R�A��л�0��� �G<x�+<�k*��`���R=DX�1�Y=�j<��#= �'�LԼ��<<q�<�J��8a`=|!���W<�q=j�=(��<����E[�(/�г������W�t0����=����=��J=Pٔ��߰� ����f��jpd��R�<�A����t=$��<�4:�N�9�z	f�'!�=��A=8Bq�e�B=N�<$j�<�V�0O�$]�����#nz=p��<C��=,����'��e�=���=�r̺�=`�+�����y�0=�ˠ�s�$=p{�<-ႽpǾ<l����a=�ƒ;@���	����; �?<P��;زJ�x�<� �q=�`R<�����O�;LU���"� T/:k�7=���}��$�J�&=t���/�=��=L�=r�=�ħ<�r����>��=�m�VC���}e��D��~��T<�0���>�����<h�=n� �<qg=�x8�Z�� J��{8��g�=�)<=x<�ƽP��;p'h<����<S�<����q�<�<��=�W7��5>P�<P������=T�ͼ�������=x�<��d����:��� �E���;<p\�L>=У<�UE�-��=T��=�=���=и5���&����(��=�ڮ=�ɼ� �һX	��{/�=������C�p��;(�����=��� O���1=��ٽ��s%=����pu�Fv=��<L�;c>=�C���s�Ht�;�������d�=�w�;8��=H��<��=e@� @A���Ҽ�gv=b*�.�#�����'��2�<�~���e��hQJ<8�<@��Bj��p�ּ���
���K�<���<D�r=�n�=�=V�=�;����k< ���x��=��z��T�·=�}1=x��=��}�N��=�3�\�<��=p1��B�=�9��i�;�� � X<@��<�aF��ڍ��~=,,�����:P���h�>��P���'��p����lG��)<֍��SS�mg=O���p�1=�SZ�CL=҉�@3������ЍN<��t<�^ּ:|��d#=�岼$�6= z�<���<��M<@-�h�=@˷:�BM��1j=�?�@�:��=�Z=��< o����7���;�K⼨����"����E�p,�=BSu=b@�M���-<0�o���Q��g�<@��:��B=x��<B�P�F���l9�*ֹ=�k4=���=|=���<Ӳ߽[Ǽp����T3;	(U=�x<ī�=�v��$s���si=rY�=po�;M�=D4��錥�8V�<>5���?=0 <.�!�h]�<tL��q"=�*a=�f�`��; (9 N�<�<8������JŇ=��W<�����;�"���<8k�<�n=̷��ߕ��g��j=�rĖ=P�p=X��<0S�<��==�+�;RO>$��<X�< �^�)���B�<<,o����=`��#
>8Ƚ7��0�=<���ğf=h!<�}���m�%c=Ј�h�n<��&;6�x�4�P��<��<X�)���:=�Y����O�a=���;�֢���U>B?���r���**=�E�a�����=X������:�㹽̨���h�<�������x��<�l���KԼ�W�=�l=�+<��=L���<=ټ�~:�E�=��= ���7���һ�Y�(��<�j?�ئ���W�=��=C��7�e�����P�"�Kɚ����}�=��=��Y��tǻ4ql������\���=��#д� C2�ȿ�#=����>f�m�X�R��h���H�=='��	-����,��~^�3�=P�ռΖ!�d=$�<l�_���=�2;����Ƚ�˝�����=�r=���=��.�T�=E�B=���*��=�dJ�[�*=��p;��,��=��=�^.=/D���s<��.�<�<`�j��2>�I����<�{�:��e<�L�< .g�ˬ� �i=8���ቼ��0W��]G�� [��Ă��|���q�=�=�2	<)���0�=O���+w= n�<t=lS��m����P��;@k�<�ր;~��Q�D=Hg���7_=1=`�<���<�����Z=�b�3N���z=x�=����;�8=��@=�UC<�����{༔s�<NH��ļA���#@�[҄��K>�=�s�����j)<D�����޹ =���<
��=��<'�����E�����{�=�g=�	[��6=H�=0��<C`���I`���<�J ����<��<ʞ�= �޻(�<���^=-�=�KH<ġ�=�#��!����Mh<�W+���^= a���z�����< ��(YN=*��=mQ<�== ���<�/�<x
H��v��=��<�u ��n'���h����<��=^��=lӈ�q�����
��3�=�C�����="%�=�U�<褱<��=@�<�>P�<��;��ʒ�����<�����
�=��!�'G=Lht���:��<v;.��D�<֟3�d���u`��x�=��z��/;��+o�u����ʽ�q=s<�]:�Գ�<��ὶ���/=j#&���f�:�/>Pe޽0:E�pbջ�������f�
=P��\��<�1ý<X̽��Z=�n���ѽ`V<4k��;���JK=���<��;��<BM��LD¼���<��=x c�8{-���0;��	��0v=^˓��s#��V>�=Sլ��%e���������#���@�W�p���Y��= �����pC�3F� VN:��=�ܽ�⩽Z~��T���<�
��Y�>HBr��x`�wڽ`A�=q亽�۟�d�缶�#��>pȏ���n�$S�<��<L�<���������W׽6=[����T��<P�=�߽Iߊ= 
6;��ȃ�=0���t�޼��hV���=:�u=�����t̽0���<��|v��{���A=F�>�W��4=@��|�<Hp<6�ѽ6��V'=F�>��>P�\kg����	���O���D�X����瞽%m=dו������=�X��dN�=$�K=V�d=��ڼBl�@���B6����<��<��n� �=����G�='�n=�=�' < o"���b=t���튒���?=<�}��ߟ:�fo=:Hn=������"��������<�3����!��hI�Q����>��s7>��=	����罰r�Jz���L��<Hʻ<B��=��<6ֽ�߅��5C����=ύ= �l�VD=���<�}@<rW"� 绺p2�;��0��D����<<3> �!�����%��=�y�=�d�<t@�=D�M�Ƚ�0��IV��S=�b��A��a=��Q�`W�=���=��<tH=8h��v�=p�<�\Ž �-�9K�=�-<����� �,ҩ��R<��=�>�=0X��������_�
נ=��C���">�Q�=`��;0�j<cϡ=��;��>4R�<0%b��9�X	t���<\��<�Ό=ʵ=�z�;�MѼ����;��q��넫�8�.�ϙ��'�h$"� 2;�硼��=��3��d$=����p���h%Y���׼��Խ�7;+�����):<.m=���\9μd�(��P��ɷ����������!�<7�����@����=����H�h{$<r�B��(��8���t+�h�ü�� ˻8F
�V�*=�F=�N��^(��/��F[<ʜ�<=<���`�h����=��<-_��k��Pu����ν@����L�};�n�>�I��=t�{��5>��ު��� ��;�W�<t��{6�y'�� �:t�׼�H���[�=@ʺXq�J��~�=�f�$�� �[���D����=`彻 ,X;����D��< T���м�����\���ȁ�`~����&�JB*=*�ӽ�҂=�P'�]V���f�=ā=���ǽt"��Q=�=��.:�d���������u�s���T�˼��=B&=P�һ��=@%�� �� ~����ڽ�>Žp�<��H�R㜽,����D�w�޽x���'�<DB��A����= S����wt�=d1��d��=h�a=�N;=������g�8�|�R�P��hH=\i�<��E��p�=��Ƽ@�=�4k=@��:�}������!�<���ne�H`�<н����[�E=��=�]���Y�x8ȼ���;�&����K��so�O齽�c�(��=jƜ=����ؽ�
�<��
��R7��E<ԣ�<Q�X= ��:$_�VLC�0����}�=�/X=�6�<P_�;�$ݻ�����J�=���Q;@  ��^Լp4���>����\�W�x��=��<<�<��j=�!(�;E�� 4��ƛ��"=����(u�<�0�;��X� 9�<0>���< ��7�WV���4=6�;��ὸ-�k��=�D� ۵� �T�H���T�7�>��=��H�?N����d��y=���tq>��=,����m;Ӱ<=8�����>0[�;ټP�P��Y��ͮ8=2�=�=�2x=�9ڼ@�z�@�ݽ�������7��j̽��7#���.��������;設��!�����h%�<�����Q���2�,u�<���M��A����.��!"������S��ă5��3��Ô�d=¼�zb����;����lF���=Л��,��(�)<��S�(o��b�2�ߵ�t�%�v�M��J�<�;�P%=�<����o
�������<�1 ����<������<L��=`�<>���� �t��� P���G��/���k�Jv�=��q�X�Z�xRJ���� �Y���
�`0<��?�ý���<��C������=`<���fh��䳽��׻V�+�?m���6<�3�<�{Y=���`Ц��J����< c�<LN��h��<�o���j���м����81����8�?=�jD��z��(b�<s,q=����鍽�.����0=0ȼ�y�����@<н|a׼�D{3����=�U�����< '<؈�8�ż̛����ɽc��pD�;&a��o��.蚽Л�<OAý�1�`�<P�׼��R�\V�=d�ڼ�K���=L������<�'�<�1=pQ�� A�;�qּ'��;�q=��;�)�V(^=�r���8S���#=����6� �ػ��a��b��
'� ,p9 �<:�J��,*=(�`= f��8`;� ���û���4n��b�����,9�i$�=�%}=�m��'��|=ȱY�t��8�n< 7:���<�ؼ�y�� ��)��E�=0�<`��;8t����߼ ��>��]���4�	�K�V���h����=�"��u��E=�$"<�B<�<gm������bM��ɽ��<��P;�<@� �������#����=(�<^�p�ͻ&=��Z�"	ҽ�1���u=^c	�����P?����x�����=#I�=@W��q��HYt�BB=LJ�����=B�=$D� ���Y�9 ���>()���m
� ̽���@C��������t�Ӽ@v>8,����J= ��=j%���;�a>=T�(b���8<��
��H!=���<e==�D��`u̼h��<�ٳ��aѼ@1.�~�C��Cv<��I=~) �=�B=�36=�ڮ���0=n��)d�,kd=��<���zp)���W����������l�<p$<Õ�����=D��=4�4��ʻlM�`�B���:�=O��=xh� >�� ~����=n`��`Y;��o������L=~
�$����K�Q����=¹~��K=.�=�s���R�<��%=�Ns;�.���w��T�5�H�y�g�І�=�������<Ȟ�<�u=�����D�@!.���=�|���hN�"��4�ؼ �ʼ��o�o񪽀���T�=t���{T�P:<�\���2��@�0��[.<`�7=Iʅ=8�6�=�Ϯ��+:�q��H��<xȼ�1� u�l=ba�=߈���x�=RdF��]=/�r=l�N�'�}=�6U;������<�c�<߁.=P�a�D�/l1=:����<T���D��<�9Խ,�� 1���ϼ$�	����<bY�h�μoo=u�Ͻ�G= ��;:�[=�!���üH�+��'����;�޼B
M��8�= Uм�\k=��<�CP=@��T��lO�<�=�Zc�sE=�������<�TC=�v�<�-=@ <�L�������<��� ���8"&��f��b�=JS=�d�s�ĽPfN�8�ۼz�0��Z�<�{��Mu=ȪA<J�/��)"���E�ER�=�+=
O�v��=��<�ɕ<7!Խ��N;zS����t��=Ԋ�<V��=d�����:�8.{=���=({.���<@T������=�s��r�=(+�<��v�X7�<Nc����i=�q��!:do��`��;P��;�e��C���A;� �=��=�p�l�<4R-��&#�P�PCi=؇ۼ�;����Ӽ=,=�, �qt�=�#�=�4%=��<(�:<L\��C>���< �{9N弽����pYY<��%�=`D����<>򮯽�%t���O=*(Z�i3=*��=����C�E��=.:U��)��P��; �m:vz罐u����#=�����h��
���}~ؽ�=�4=�e�|h3>�D���Ͻn=�o�`t�m�=X�1� ;�:�.ͽ5����R��R��`
�\q�<���<������=�'�= ��;��<�p��|��B����
>�̰= �,��h+��<���;�<@��Оڼ�Է�x�<��=l� �F���� 4����;����h�~<<�=�!��
�<@�1��:�9���.gp�^�=w���`��.C�=\�K��,�=�p�<��!>mE��|���t��J�=BsH���v��]�o掽��M=��6���� 7K;�=HcE�D2
�4M������p���༖�=_��=��;�<s��=��a>C=�怽��=\�@�E�9�5=>��=���=y�Ͻit�=�.��"n�= [=�}z��A>�0���{��qh<���<���<�⸼�`��.=��N���";@�\G�<��r���R�\����\�����?�<�z���ƭ�e{+=c妽t�
=�$�;'=�6�1q��H�(�� �;�v�@�¼4�.=Зݻ��=%�<ԯ=�#H<��w���2=���<� ��EC=�II��A�;�F=�s<j�	=�I�:�O���,<`��;�Q��8����<b�,���Nh�=�Q�=XM�Z�t�PeĻ���"�
���<��+���= �;��o������=^^	=\?��K�d=��< r�<����� ��l�� >��K�R=|��<j;�=�\��xVü@2�<T��=�0Ǻ��~=H�[��H��H=\"����:=�͒<<�ۼ�4�<�}C�3<=(�=x�S<(�b<�y<p�<@���`���p� ��a�=:=q#�h]K<�y޼櫼0�<��y=���~E_�0�^�B�f=p�m��k�=8��=���<��<�=<�$�=H��<��<�T���7����=諒-a>�B��4�7>��ĤW��z�;�y��j�=X��<J���!��GV�=�7��<{�x~
���������9�<�/=(^C��XL<73�f�@�(�=�W�<��;؍>����Ž؆�<�I����G��F�=2i��w{=���|�PE��t�Ҽ�
�� vT;`�<(�<�z�=Ao�=�f=�M;��H���޻�^H�9�+>��<(j2�X���P������;ڌ=�=���|���>�sp=�2�V�����~�r��q�V��د���C�=ܘ!� $o��8I���D���˽J�/�r�>�N�IC������NԷ����=�ī��Y>ya��`�&�L���?�!>�L˽W���fNq�'�ν��>p�Ի&���<���<Nx���]G�B(���25���A���={`&>���ԉb=ͱ`=V�H����=L9��E�=��R��&=/��=N�>5m_=r���W�<	��e��=`;(<�9����>����t�Hl���<~�=�.Y��C ����< �˻�Ơ�X{�h�<t:���;.<)���;@���8~=c<6�}Iv=����Z/-=���<� =m�����;%� ����͍;x�'<����u"=��;(�<sJ5=Ь�<�C�<p�����=xx7<8ȯ��g=� Ἰ�%<&z�=�<� �<(�q�8<�&�<�S �`ʌ; ��k��j���
�=§=S����>��݂�lG4�H)j�EHD==e<h�<P�D�����T<� J��$�=�M�< �`<�OH=Gb;=D��</Q���ᵻ��A<��p��<���<*S�=��|���'���K<3��=@��:��=�z��~e��?�<ް�7�m=@��:�V�:@=�żw�7=�=�=�=#=�%2<|k�<@<<p���ر����=�F�<���Xv��1��`P��Ιv=P��=�{���0�x�b�*�=��;���=��Z=���<�0��
i=�*=F3�= �I<T5�<���N���/=`
��->h-)����=�������d�׼UL���3[=��T(�QS���p>�����:��h,�[;Ͻ��׽�K}=�ȴ;�Ix�ȂA<�d��R��$�=09�LK�<�<t>�B�`ϕ�	���=H��@�pY�<ďM��N�= H�{
�ȯ~<��!��_ݽ�O����f�<���=n�~=xt�<�����(�  �`%�;�	>�0���K�
���+J�TaB�f��=2����1Ž�A4>`��<oͽ��r�7����s^��,���|��H#i��
=-�<T1F�4�����5���쳼5">��-��ț�AH�����=3��4>�τ�����	��>����2��޷	��/���(>�Γ<�N�:�=��&;��{� �"��%ֽ93���ẽ�N���M=�&>�h7�I��=�
b<h�5���>�~[��E�;��]���<�u�="e�=�!<����ڼ�㛽�7;< K����y<�qg><������������=@��<��B����Y�;Ǽzz��_h���2�FZ��K$;�`���*�����X= ��㐾��=b/a�ûy=_�9=Hw�<8fp�xQ��O8��0+� �<8��<p��b= �;	=Ƅ�=��<��<4���͉=x���]�9=ؐ����;���=d��<  �:Ȥ�إR<Xa�<��&� ���P���n�/�&�N�&��=k�=��ý�⇽4��No������.=d�<�x>=��ż�5������`*<�d�=h��<t4�<�G1=��=���<]Dٽ�F3;d��<��޼@>ں��t<���=0��,���,�<�[�=���;^��=p�ļ�c��Y��WU��`V=h�A��؝<� =4(�n�T=���=_L=d��<�V��(��<HEn<BC������7F�=�3�<�)�<a#��Lt�X�8��l�=�Qw=H߼j�b�,���ۋ=�{S<D�="��=�	�p_�����=&x=.�=�uH;X�<�����f�'�<x�=<���=�!�<7_0=�3���K��ż�낽�lA<CZ������l�z�=�D�"] �dd=�sB��Vg��xe�=�T�@�,� `���ue�F!��<ޖd��L9=��=D����������.����� ׻�&�a9=:���
���-=�Z������Լv�� &t9��]<�OL<4̍�&�o�
x���u_���<}�=�pW� �X��`��6��f2�l;�= t���a��<�>������l�$j�F{��K�6��B� �D�>��巼��=^����T���m������g�:J4�=�e��$���5��d�=���̼�3��/��=8��<�����ҽ��o=N�a�����H�[�2N�V�
>\"�<`lU<�|�<`�R<hs�����$Qż�BȽ����.������=�{"�+S�=:o ��޽��=�ǰ<��ƽao�$ּp��=�m�<T@；���8�|���E�O%��p񱻢��=ӓ�=`�T��낼h���5�;hC�0�[��Fg��PD���¼ dýlכ�Ԋ�<�@����@�<pf��Ă���|�=0�9��T��5�@=xz�^�%=8�j=|�< ���l��<�ݠ�'ۗ�?=�,�<���6�i=� �: �?;�΂=`�� » �R;�Z�<�B!�hn����c<�C=<|;��_�e=f-<=@�X� 6����;phe<bp�|��0�'�Z�q�4���
7=C��=+½&�K�@p�;X`ܼ �};���<���<(q<��;�}س��X� ꚺ�D�=��<8�R=`�<� ���@@;9a½���0E�<p�ܼ�����弉U�=����(����<L��<8�n<
T[=:��=閽$Cj��맽S�=��a;٪{= r�;�:	��D<J�>�L/=@��� -���=f*<d�����μ�8�= ����~�4)i� ��|�üx�>nYx=�Z���SC����<=�6<<�=ƭ[=�ͼp��/=�S�;�f�=�}���4�;d+��0an�6�"=�R�<%x=�-=1�<�g�,U%����ʎ�0Ή�5����2C�F�i���==��ƻ0�6�t�V�}j��4R�K�o=|���/��zȼȐX�����@@,;]z��df0=Ui=���@Nܼ(;����z����^z��V�i�=qa���{�A=�*��B��LoҼ�8_� �=�`0� (���#�a7���b���ټ�6O<�q=�Ck�(�C�،'��ch���%��;=�x��T��/�=0�6��c� ρ�vt@�!���L�����H� �6##���=ơ���?��@���������P�<� FV���X��v�+���ƽxq�=�Tj<@��r��`��<Jr��ϙ���Ѻ�iǼ�L�=ج�<`�v;X�[�$;�<p�;�'i����;7���ټX�%��.l��T=л��'c=�@H��謽�Ǖ=�/=����c۽p�$4A=�<��/6����:����zO���Ā��DJ�=h��<H�}<�S� �7�
>�0����;�������ԙ��<���`���	�)=X����2��'?=��B�0���V=*��O5�� �>� t<�'Y;���<0ݻ;𽅻q�C=\���νlV= ��9����%= ���� �;�=\��@%��_�<H_��?�;�	�:��`4=�� �}0=�
= ���ܠ�<�!��K��P�ϻx�����=���k�(� <�<<�RR=����z5�&T =���;�{׻X$3<���;Ȑ��@���z��8A���Ǽ��0=*:�y�=�B���Ǽ@����7���W��@ö�@��(L��fȇ�-��=:�&�ؚ<|�<�V뻰��;xw�<�|>��r�����=�Ľ�͓< ��;W�=t_���M��G򼫠�=�S�<\�#�.�; = d�9���ͼ���=t�ȼ@h,��Q��O<��ӼKy�=�Iq=�/��^�?����=�<���X�#= T=�X���ؼ����l����&�=�������n�̯�� �c�����yN<�����=���8��<���<n�� O=����<���(��=��y��웼 ̓9:~&=� ��~�a��	=�� ���LR=��/���N=D��<R�`���5<�ƕ<�`-� %�9�f�L㼘�U<�Ć<@�r<�g����=���L>(���3<�~<ȯp<g+��m=�p�=h<�]*��都T1�<geֽv��=�;�=�\��K�֥���}�= �b� I;�҉<p�v���r<?�̽��.���z��'���T=]#��14L=�=4�>�|.	=pWK<��=0���������5��\���O�=܃ ���!���X:ѹr=/H޽�,�`-6;l��<�Z'� rI9Tݼ�\��nom���`��� �z���F=>�x��� Y�;6�+���Խ�P2�NF=��=���<DC��� <�����/�<���=d󎼀*<����y=�=n`�"{�=eI��z�Z=)�=�ys��gQ=���Hz'<3:'=O�5=:�[=h�;�����_b=Ĉ����X<l"ƼǠ2=�� �ռ�]��<M���q���c>=�@K� �� �<CY���]C=��J<(vW=r������ �i�(fS���;pZ�~��rώ=hDq�@�v=���<�g=��e:�3��=bY*=�!�`�?=އ��<��<�e=��X;�]4= `�����:�K,�g�G=�)����f�@�M;�I{�*y�=�H=VZ�[�ŽPQּ���+Ҽ�[	=$벼�n=����rO�|�ͼ,��,��=Z3=�1"�'��=0Ǥ<�*�<}�����<Lϥ�,���9�=��<K��=x�'�\�I�?	Z=�|�=�D0�8/\<�-t<�i��r�=hfx�XG=6=�kK��=(i����f=��o����<�|U��<P��;0]a��#'�H}<�Ŗ=�$=`��;<��<���>|=���&��k=�N|�{х�������<����$�{=`��=G�:=��W<Pg�;)���S>^�<���;@GV����$<�u����=�L�[|$>\5��28�� �T9Bo����<>�8=��G�>�|�_��=O|��;x�����@���@$��vA�ȦS= �g��&μ4���Ƚ��=8��<ҷ&�AX>�[j�p�{�_�<ƦX���۽̚�<8���as=��Ͻ�н��� E�@B#� d<4=�< �����=��=�i:=���e�&���<���0;>{}\=�T.�`s��$x���}�:��y;4�F*:�A�<x|.<U#����6�BZ�l[:�0ǆ�������<ҭ�=�e何b =��ȩ�<V,X�/鍽l^�<O��G���p=�z��r�9=�V��r>cz�p�~^���Q�=�R`�~R�>_C��Q��x�z<��	��#��Y��=,v�Ш���:��=r�����7<�N��=(��=�J�T ����=.���܌=��A��w�=4#���ni=�n��8��=;�=��½�[�=�����a�=��=�����s<>��� ����h=O=���<�Sټ0�e�h��<H�����;:�U(�w=*%t�h'C�\Ս� jһ�����{=X�	�`�;<�<_O���= ��<
�=P�ʼ=ؼ�,�dyƼ�����J<ȣu�ga_=�V<;�w*=�r=��)=8�y<Z �^D@=lN�<�t=�<C�Ђ�<	�k=�ǈ:��=�A
��"�<���<���<�R>;L� � ��<�¼�3�=��~=j�f�����Q���N��^y���
=�|��-v<=@ܼ�,��^��@L8����=�v=8��=���<L��<_�ߜ<̕-� ��:J8=Ӡ=�D�=�r!�T{#��ې<ַ�= Ҽ�;=02�;���c�=lG��H�(=�F�<d葼r�#=��(��$K=���<h��<�c<@��;��<0鮻���7�;���=i�4= �E9��<�����1�H�#<L=4O��ʌ"����Ks=��K��l�=���=�`�<@�;��=��<]w�=�Ȣ<t��<@T漪����M�<hgʽ.�>��->(��<qm���6������!?=df�<��K�a���Zd>ހi�����K��m�����ǽ�Fo<	�D=t-���;�)E���&���>\�<@X����s>�!	��d����9;U���$�+�jT=�w����=6��j�	�*	�H#��0^�����D�<C�=.��=��= g=�,μ[�c�(�<~h�t�3>���� ��';�ں�]�轚�=��tS���>��;#�X?���H��Qg�k���;��T.�����=����@<�F���ʷ����2vt�q>na��Eg� '�9�3����=l����E>X�0[ݼ���>Y�/
����d����5�=���;@)2�ϧ=�;�ٜ��[�;P��O뽺[��x��+l�=�<&>^�&���j;T�=P�>�h�> -A�9^�=^i1���=��+<�\)>X ?=T��h�<Ѝ��S�=@ku�P���^��>�Eཬ,ü�K�:�=P
=`��h�z���;�=�|���h� ���<LO���7'<|���@�{<�軆	= �?<pk3�=��e���<w�<�o<��^�hSV�XA!��#�0&����< 7;�=<��<���<e�+=�<�a�<�ۊ�$\�=�T�<� t�m4+=蘺���<��=�Kp�$S�<htj�X�=� =l��<8��<��� U�:��Ǽ��]=��=g:����%��`�P�Q��慻�:=4G�<h��<���t��z�(����<	)�= G9<$�<Agk=?7'=C�<BA���s<��ǻ@(ݻ�ޕ<��<��T=�NU�,/�Hς�Y$�=`D�K�w=pt=�@f���<=��QG=`!�;@��;1�-=�C���*=Q݇=��U=���< �D<,�<�L�;X�ȼ�7i:b�=�
=���;`�0< _��������0=�w=x�N����:��E�U=\#�<�W=�Td= d)<����LA=ښ.=��f=�' <��<�*��T�T�<p�X�h|>6Z�����=d�ʼ
d��띇��V����=p�3��l��r�<�P� >���N�`���������R=�s�;|�d�8H|<m�.�§4���=,o�<�2�<�5X>���\�"�̈��K��&����<0U���=V@�Z^� 􀹠ѼH���n-��إ;FA=�Ss=��{=Zf=�n�I_=�2=��!<���=s"��1��=�T�����$�;_�=dxQ�N0��+#>P~5�����)�f #��iI��/��J _���Bs$=��R��@'�2����j�������;���9>QZ:��A���x�x���=��f���>��"<�裻s 
�B��=�ֽv<|�bf���ݽ�>
N=h�<M$u=l:��׎��ں;p�ٽ�ܽ1e��n�����=��*> �T�N�	=��N;R��>xd�P`x=�4�&�=�/�<���= r�92���r���Gֽ�0'=�S,�Dݽ��$j>�Q��\ȼ@����	=�L�<����:�4ݪ�X��jW��R1�03<K�����;D��h�<HC[��
0=��;V�W��L-=D�8�=�y(=pd�;d��#���a3�:/f��p��Z�< 6-��]=@~y<���;�f=x��<X�<�J���<�=�l6:p�5��&=��� �l<i��=�K;pY�;���n=0?�< �.��[T< Ha��X�f����P=�Y�=�Ӿ�f[F���X�n?�� ��?%3=�A�<��<�2��v�Hb��(�<޲= 8�;JK=k�;=*$=���<��t� �Y<���;<+Ѽ;���}1<��h=�# �|�����ʼG��=@w;��M~=�����:I�`CQ;��*�F�"=�޻PQ�<��=�H��x-=�x�=�̀=xr�<�$<���< \<\Z:� jm��G�=�<h�L�`�q��K.�^���1�=�c^=�8��J������Вg=d��<��l=��f=p�V��=8�|ln=�(=�mb= 'c�dx�<@tq;\�X����; C�:��=L����%=�E����b���`������A<�U�� ��@����=��"�՝����r�r!���/��$�=��#�(|)�P`<�Ƚ��&�R=��ѻ<=C^�=*�۽`Tۻ�̒�2�L��8�8�h�0����=�󤽾6����<(�~����A�� �ȼĺ�<��<�-<����<;�Nǽ��<��<.�e=H�U���L���콰	��
�T��=���D�]�=�!���5=�d������  �-L�̨ȼ\	��0����=�*��2����Tٻ]bŽ���R�=rc��"�eν�߮�d�������K=�S,=@\X;~W�1DU=o;��r��:<����(|>��D=p�=�P=�k�{�8����̼u���XH���!��`]Z<}8�=r�4�F==��}����8�=���N���齘O�<L�<���<F���_��P�K�������ؼp�9��V+=y��=���E���=-<��W�9м���R{���s��ެ�P�|�hn�<�?޻����0f�;�D���^��~F=��������q<,Є�tĻ<Bmc=���� �1: �k;pvl�yݟ� {6;8��<�$L�T6=Є&<�n6�;T= >� ^Q9@`���=@�ܺ��>;���;�F�;P�	�i�N=���<���;�}C����<$�<�8� h�;8�����a���q�̘�<�DZ=�ɽ�8������� ���lS�<h�k<�w�;�Ќ��Q�z�	��[-<،�=��;�3I= ��; �P���;�]� ����:�<BܼH3}��4���q=X= �HO4��徼��O=<��E=B޼�0�@��{��4��< 
�;B�u=��\<t��xJ<7��=�N=@���Ծ;���<�!<;��m��ﺼ��=@>�;���Xe��e׼��(�@$�=�M=4:��Ή� ����S=@ك<�9=;98=f�����J=q|<��t=�j�H*d< �;�>S�`�J<X��<�lN= 9���^�<����:�.;V��ʝ��;�#��h�)� �D�=^�=@ƺn\
��:��+��hJ�Ŭ�=�k����� [+;��j���ܽ��=��ǼޡN=˃=�������ј�v�'�c3佸#T�P�׻5zs=������@�+;���&����\��^��@��:xX��E�;�����p���r������<�q�<$h���<�a���|�T�~O:=ؙ^�j���O��=!ؼ��ͼ��&�����ɽ��� |��8��DS��n�S=b���>|��P��;��˽����d�d=�i�`�<9Ľ �2�~)(�í�DB�<n�= Y�Z��\(�<�~!��8�����,w�p��=�#=`��<�y<�"�:�<���m��A0;�^� �������#��=*���p<�<V�:��M~�8��=�@[;�;��>֙��E�:��*<����:����$t��?��V�/����<u�=�,= ົ�^��Q!�@�R��m�� iM��tD��./�pB��j򭽔Q\�C=�< z��n�<pA�ح�<L�<05�;�x��^��S�<8���=�됼���;�]%=p�߼�Iǽ�Ϸ< N79��);0$< �;�H9����<��!��v��X=��`�;,=�I���E=D:��<P�<�7�; (�<H�W<`�%�X�u<xI<��Hֻ`T�<�����<���p �(�<t�<�U���-< Q{;&�������I#�6�;SZ�xa�<�Lؼ��+=��P-��,�:�����&7�@�G<�a�*��fi���<N���)�;D��� ��9$���`�O;n�!��ּ�k��g�� �8H<D*�=܆��h�����"�=�ζ<H,��7<�U�<@2��p��xJü=�n=�W�0!|�������<����K�=7g =���|[ż�����~�;8%<��� }�:·�*��P�#���wC)=䄄�Xw<*
dtype0*'
_output_shapes
:�
�
siamese_4/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*
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
siamese_4/scala1/AddAddsiamese_4/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_4/scala1/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_4/scala1/moments/meanMeansiamese_4/scala1/Add/siamese_4/scala1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
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
 siamese_4/scala1/moments/SqueezeSqueezesiamese_4/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
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
Dsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
 siamese_4/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_4/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
(siamese_4/scala1/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0
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
Tsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
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
Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Lsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_4/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_4/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
c
siamese_4/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
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
siamese_4/scala1/cond/Switch_1Switch siamese_4/scala1/moments/Squeezesiamese_4/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_4/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_4/scala1/cond/Switch_2Switch"siamese_4/scala1/moments/Squeeze_1siamese_4/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_4/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_4/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_4/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
siamese_4/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_4/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
�
siamese_4/scala1/cond/MergeMergesiamese_4/scala1/cond/Switch_3 siamese_4/scala1/cond/Switch_1:1*
N*
_output_shapes

:`: *
T0
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
 siamese_4/scala1/batchnorm/RsqrtRsqrtsiamese_4/scala1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese_4/scala1/batchnorm/mulMul siamese_4/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese_4/scala1/batchnorm/mul_1Mulsiamese_4/scala1/Addsiamese_4/scala1/batchnorm/mul*&
_output_shapes
:{{`*
T0
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
siamese_4/scala1/ReluRelu siamese_4/scala1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
�
siamese_4/scala1/poll/MaxPoolMaxPoolsiamese_4/scala1/Relu*
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
siamese_4/scala2/Conv2DConv2Dsiamese_4/scala2/splitsiamese_4/scala2/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�*
	dilations
*
T0*
strides
*
data_formatNHWC
�
siamese_4/scala2/Conv2D_1Conv2Dsiamese_4/scala2/split:1siamese_4/scala2/split_1:1*'
_output_shapes
:99�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_4/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala2/concatConcatV2siamese_4/scala2/Conv2Dsiamese_4/scala2/Conv2D_1siamese_4/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:99�
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
siamese_4/scala2/moments/meanMeansiamese_4/scala2/Add/siamese_4/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
3siamese_4/scala2/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
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
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_4/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_4/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Dsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_4/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Hsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_4/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
Ksiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Jsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_4/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
siamese_4/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_4/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese_4/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_4/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
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
 siamese_4/scala2/batchnorm/RsqrtRsqrtsiamese_4/scala2/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese_4/scala2/poll/MaxPoolMaxPoolsiamese_4/scala2/Relu*
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
siamese_4/scala3/AddAddsiamese_4/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:�*
T0
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
"siamese_4/scala3/moments/Squeeze_1Squeeze!siamese_4/scala3/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_4/scala3/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9
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
ksiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
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
Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Fsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_4/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_4/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
"siamese_4/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_4/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
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
siamese_4/scala3/cond/switch_fIdentitysiamese_4/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_4/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
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
siamese_4/scala3/batchnorm/addAddsiamese_4/scala3/cond/Merge_1 siamese_4/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_4/scala3/batchnorm/RsqrtRsqrtsiamese_4/scala3/batchnorm/add*
T0*
_output_shapes	
:�
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
siamese_4/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_4/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_4/scala3/batchnorm/add_1Add siamese_4/scala3/batchnorm/mul_1siamese_4/scala3/batchnorm/sub*
T0*'
_output_shapes
:�
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
 siamese_4/scala4/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_4/scala4/splitSplit siamese_4/scala4/split/split_dimsiamese_4/scala3/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_4/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
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
siamese_4/scala4/Conv2D_1Conv2Dsiamese_4/scala4/split:1siamese_4/scala4/split_1:1*
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
^
siamese_4/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_4/scala4/concatConcatV2siamese_4/scala4/Conv2Dsiamese_4/scala4/Conv2D_1siamese_4/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
%siamese_4/scala4/moments/StopGradientStopGradientsiamese_4/scala4/moments/mean*
T0*'
_output_shapes
:�
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
!siamese_4/scala4/moments/varianceMean*siamese_4/scala4/moments/SquaredDifference3siamese_4/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_4/scala4/moments/SqueezeSqueezesiamese_4/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_4/scala4/moments/Squeeze_1Squeeze!siamese_4/scala4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Hsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Esiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_4/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Jsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_4/scala4/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_4/scala4/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
usiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
"siamese_4/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_4/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
siamese_4/scala4/cond/switch_fIdentitysiamese_4/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_4/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_4/scala4/cond/Switch_1Switch siamese_4/scala4/moments/Squeezesiamese_4/scala4/cond/pred_id*3
_class)
'%loc:@siamese_4/scala4/moments/Squeeze*"
_output_shapes
:�:�*
T0
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
T0*
N*
_output_shapes
	:�: 
�
siamese_4/scala4/cond/Merge_1Mergesiamese_4/scala4/cond/Switch_4 siamese_4/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_4/scala4/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_4/scala4/batchnorm/addAddsiamese_4/scala4/cond/Merge_1 siamese_4/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_4/scala4/batchnorm/RsqrtRsqrtsiamese_4/scala4/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese_4/scala5/split_1Split"siamese_4/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_4/scala5/Conv2DConv2Dsiamese_4/scala5/splitsiamese_4/scala5/split_1*
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
siamese_4/scala5/concatConcatV2siamese_4/scala5/Conv2Dsiamese_4/scala5/Conv2D_1siamese_4/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
score_2/Conv2DConv2Dscore_2/splitConst_2*
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
score_2/Conv2D_1Conv2Dscore_2/split:1Const_2*&
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
adjust_2/Conv2DConv2Dscore_2/concatadjust/weights/read*
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
adjust_2/AddAddadjust_2/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0"��=