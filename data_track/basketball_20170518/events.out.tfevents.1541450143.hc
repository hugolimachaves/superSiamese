       �K"	  �g*��Abrain.Event:2�7�.[�     Զ _	�q�g*��A"��
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
<siamese/scala1/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala1/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
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
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`*
use_locking(*
T0
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
,siamese/scala1/conv/biases/Initializer/ConstConst*
_output_shapes
:`*-
_class#
!loc:@siamese/scala1/conv/biases*
valueB`*���=*
dtype0
�
siamese/scala1/conv/biases
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *-
_class#
!loc:@siamese/scala1/conv/biases
�
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
shape:`*
dtype0*
_output_shapes
:`*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma*
	container 
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
/siamese/scala1/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
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
siamese/scala1/moments/varianceMean(siamese/scala1/moments/SquaredDifference1siamese/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
siamese/scala1/moments/SqueezeSqueezesiamese/scala1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:`
�
 siamese/scala1/moments/Squeeze_1Squeezesiamese/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
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
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape: 
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
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readsiamese/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
�
7siamese/scala1/siamese/scala1/bn/moving_variance/biased
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
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read siamese/scala1/moments/Squeeze_1*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivGsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
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
siamese/scala1/cond/switch_tIdentitysiamese/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala1/cond/switch_fIdentitysiamese/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

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
siamese/scala1/batchnorm/add_1Addsiamese/scala1/batchnorm/mul_1siamese/scala1/batchnorm/sub*&
_output_shapes
:;;`*
T0
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
VariableV2*
shape:0�*
dtype0*'
_output_shapes
:0�*
shared_name *.
_class$
" loc:@siamese/scala2/conv/weights*
	container 
�
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�*
T0
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
,siamese/scala2/conv/biases/Initializer/ConstConst*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala2/conv/biases*
valueB�*���=*
dtype0
�
siamese/scala2/conv/biases
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala2/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
siamese/scala2/splitSplitsiamese/scala2/split/split_dimsiamese/scala1/poll/MaxPool*8
_output_shapes&
$:0:0*
	num_split*
T0
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
siamese/scala2/split_1Split siamese/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*'
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
\
siamese/scala2/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container 
�
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
-siamese/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/meanMeansiamese/scala2/Add-siamese/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/mean*'
_output_shapes
:�*
T0
�
(siamese/scala2/moments/SquaredDifferenceSquaredDifferencesiamese/scala2/Add#siamese/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
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
siamese/scala2/moments/SqueezeSqueezesiamese/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
 siamese/scala2/moments/Squeeze_1Squeezesiamese/scala2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
<siamese/scala2/siamese/scala2/bn/moving_mean/local_step/readIdentity7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readsiamese/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMul@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biased@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/x@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
VariableV2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read siamese/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
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
Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
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
siamese/scala2/batchnorm/mul_2Mulsiamese/scala2/cond/Mergesiamese/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
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
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<*
dtype0
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
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
siamese/scala3/conv/weights
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��
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
siamese/scala3/bn/beta/AssignAssignsiamese/scala3/bn/beta(siamese/scala3/bn/beta/Initializer/Const*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
/siamese/scala3/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/moving_mean
VariableV2*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
"siamese/scala3/bn/moving_mean/readIdentitysiamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala3/bn/moving_variance/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*  �?
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
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese/scala3/moments/varianceMean(siamese/scala3/moments/SquaredDifference1siamese/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
siamese/scala3/moments/SqueezeSqueezesiamese/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zeros*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
>siamese/scala3/siamese/scala3/bn/moving_mean/local_step/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepIsiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
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
Asiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
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
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
VariableV2*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala3/siamese/scala3/bn/moving_variance/biased/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zerosConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *    *
dtype0
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
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala3/batchnorm/addAddsiamese/scala3/cond/Merge_1siamese/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
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
seed2�*
dtype0*(
_output_shapes
:��*

seed*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
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
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��*
T0
�
<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *o:
�
=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala4/conv/weights/read*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: *
T0
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
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala4/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*-
_class#
!loc:@siamese/scala4/conv/biases*
_output_shapes	
:�*
T0
V
siamese/scala4/ConstConst*
dtype0*
_output_shapes
: *
value	B :
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
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
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
\
siamese/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/concatConcatV2siamese/scala4/Conv2Dsiamese/scala4/Conv2D_1siamese/scala4/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese/scala4/AddAddsiamese/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
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
VariableV2*
shared_name *)
_class
loc:@siamese/scala4/bn/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/beta/AssignAssignsiamese/scala4/bn/beta(siamese/scala4/bn/beta/Initializer/Const*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
VariableV2*
shared_name **
_class 
loc:@siamese/scala4/bn/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/gamma/AssignAssignsiamese/scala4/bn/gamma)siamese/scala4/bn/gamma/Initializer/Const*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala4/bn/gamma/readIdentitysiamese/scala4/bn/gamma**
_class 
loc:@siamese/scala4/bn/gamma*
_output_shapes	
:�*
T0
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
$siamese/scala4/bn/moving_mean/AssignAssignsiamese/scala4/bn/moving_mean/siamese/scala4/bn/moving_mean/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(
�
"siamese/scala4/bn/moving_mean/readIdentitysiamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
VariableV2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
siamese/scala4/moments/varianceMean(siamese/scala4/moments/SquaredDifference1siamese/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
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
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biased*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Isiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *    
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
>siamese/scala4/siamese/scala4/bn/moving_mean/local_step/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepIsiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
<siamese/scala4/siamese/scala4/bn/moving_mean/local_step/readIdentity7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readsiamese/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
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
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivAsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivGsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
siamese/scala4/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala4/cond/Switch_1Switchsiamese/scala4/moments/Squeezesiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*1
_class'
%#loc:@siamese/scala4/moments/Squeeze
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
N*
_output_shapes
	:�: *
T0
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
siamese/scala4/batchnorm/mulMulsiamese/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
siamese/scala4/batchnorm/mul_1Mulsiamese/scala4/Addsiamese/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
siamese/scala4/batchnorm/mul_2Mulsiamese/scala4/cond/Mergesiamese/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
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
VariableV2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights*
	container 
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
 siamese/scala5/conv/weights/readIdentitysiamese/scala5/conv/weights*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
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
=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala5/conv/weights/read*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
6siamese/scala5/conv/weights/Regularizer/l2_regularizerMul<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
,siamese/scala5/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala5/conv/biases*
valueB�*���=
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
siamese/scala5/splitSplitsiamese/scala5/split/split_dimsiamese/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
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
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*
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
T0*
data_formatNHWC*
strides
*
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
3siamese/scala1_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese/scala1_1/moments/varianceMean*siamese/scala1_1/moments/SquaredDifference3siamese/scala1_1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala1_1/moments/SqueezeSqueezesiamese/scala1_1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese/scala1_1/moments/Squeeze_1Squeeze!siamese/scala1_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
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
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese/scala1_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
 siamese/scala1_1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
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
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese/scala1_1/moments/Squeeze_1*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
�
Tsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Nsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
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
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
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
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese/scala1_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

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
siamese/scala1_1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/MergeMergesiamese/scala1_1/cond/Switch_3 siamese/scala1_1/cond/Switch_1:1*
N*
_output_shapes

:`: *
T0
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
 siamese/scala1_1/batchnorm/RsqrtRsqrtsiamese/scala1_1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese/scala1_1/batchnorm/mulMul siamese/scala1_1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
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
siamese/scala2_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese/scala2_1/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
3siamese/scala2_1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese/scala2_1/moments/varianceMean*siamese/scala2_1/moments/SquaredDifference3siamese/scala2_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese/scala2_1/moments/SqueezeSqueezesiamese/scala2_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
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
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
Nsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
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
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese/scala2_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
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
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala2_1/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/Switch_2Switch"siamese/scala2_1/moments/Squeeze_1siamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese/scala2_1/moments/Squeeze_1
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
siamese/scala2_1/batchnorm/mulMul siamese/scala2_1/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese/scala2_1/batchnorm/mul_1Mulsiamese/scala2_1/Addsiamese/scala2_1/batchnorm/mul*'
_output_shapes
:99�*
T0
�
 siamese/scala2_1/batchnorm/mul_2Mulsiamese/scala2_1/cond/Mergesiamese/scala2_1/batchnorm/mul*
_output_shapes	
:�*
T0
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
siamese/scala3_1/Conv2DConv2Dsiamese/scala2_1/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
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
%siamese/scala3_1/moments/StopGradientStopGradientsiamese/scala3_1/moments/mean*'
_output_shapes
:�*
T0
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
"siamese/scala3_1/moments/Squeeze_1Squeeze!siamese/scala3_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
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
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Nsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
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
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
 siamese/scala3_1/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese/scala3_1/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
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
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala3_1/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
g
siamese/scala3_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala3_1/cond/switch_tIdentitysiamese/scala3_1/cond/Switch:1*
T0
*
_output_shapes
: 
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
 siamese/scala3_1/batchnorm/mul_1Mulsiamese/scala3_1/Addsiamese/scala3_1/batchnorm/mul*'
_output_shapes
:�*
T0
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
 siamese/scala3_1/batchnorm/add_1Add siamese/scala3_1/batchnorm/mul_1siamese/scala3_1/batchnorm/sub*'
_output_shapes
:�*
T0
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
siamese/scala4_1/concatConcatV2siamese/scala4_1/Conv2Dsiamese/scala4_1/Conv2D_1siamese/scala4_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
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
!siamese/scala4_1/moments/varianceMean*siamese/scala4_1/moments/SquaredDifference3siamese/scala4_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
&siamese/scala4_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
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
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese/scala4_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
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
 siamese/scala4_1/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
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
usiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
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
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
g
siamese/scala4_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala4_1/cond/switch_tIdentitysiamese/scala4_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala4_1/cond/switch_fIdentitysiamese/scala4_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala4_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala4_1/cond/Switch_1Switch siamese/scala4_1/moments/Squeezesiamese/scala4_1/cond/pred_id*3
_class)
'%loc:@siamese/scala4_1/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese/scala4_1/cond/Switch_2Switch"siamese/scala4_1/moments/Squeeze_1siamese/scala4_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala4_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
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
siamese/scala4_1/batchnorm/addAddsiamese/scala4_1/cond/Merge_1 siamese/scala4_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala4_1/batchnorm/RsqrtRsqrtsiamese/scala4_1/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese/scala4_1/batchnorm/subSubsiamese/scala4/bn/beta/read siamese/scala4_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
siamese/scala5_1/splitSplit siamese/scala5_1/split/split_dimsiamese/scala4_1/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
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
siamese/scala5_1/split_1Split"siamese/scala5_1/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*'
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
score/transpose	Transposesiamese/scala5/Addscore/transpose/perm*
T0*'
_output_shapes
:�*
Tperm0
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
score/Conv2DConv2Dscore/split_1score/split*&
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
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*
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
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
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
score/Conv2D_4Conv2Dscore/split_1:4score/split:4*
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
score/Conv2D_7Conv2Dscore/split_1:7score/split:7*
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
S
score/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
0adjust/weights/Regularizer/l2_regularizer/L2LossL2Lossadjust/weights/read*
_output_shapes
: *
T0*!
_class
loc:@adjust/weights
�
)adjust/weights/Regularizer/l2_regularizerMul/adjust/weights/Regularizer/l2_regularizer/scale0adjust/weights/Regularizer/l2_regularizer/L2Loss*!
_class
loc:@adjust/weights*
_output_shapes
: *
T0
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@adjust/biases
�
adjust/biases/AssignAssignadjust/biasesadjust/biases/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@adjust/biases
t
adjust/biases/readIdentityadjust/biases*
T0* 
_class
loc:@adjust/biases*
_output_shapes
:
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
adjust/Conv2DConv2Dscore/transpose_1adjust/weights/read*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
e

adjust/AddAddadjust/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:,*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights
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
save/AssignAssignadjust/biasessave/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@adjust/biases
�
save/Assign_1Assignadjust/weightssave/RestoreV2:1*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights
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
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2:7*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`
�
save/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave/RestoreV2:8*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
save/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave/RestoreV2:11*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
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
save/Assign_13Assignsiamese/scala2/bn/gammasave/RestoreV2:13*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2:17*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�*
use_locking(
�
save/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave/RestoreV2:18*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2:23*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma
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
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2:26*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2:30*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2:32*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(
�
save/Assign_33Assignsiamese/scala4/bn/gammasave/RestoreV2:33*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma
�
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2:34*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_35Assign!siamese/scala4/bn/moving_variancesave/RestoreV2:35*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
save/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave/RestoreV2:39*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2:42*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
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
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*&
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
siamese_1/scala1/AddAddsiamese_1/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
�
/siamese_1/scala1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
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
!siamese_1/scala1/moments/varianceMean*siamese_1/scala1/moments/SquaredDifference3siamese_1/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_1/scala1/moments/SqueezeSqueezesiamese_1/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
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
ksiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
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
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
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
 siamese_1/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_1/scala1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9
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
Nsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_1/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese_1/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_1/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
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
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*&
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
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_1/scala2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
siamese_1/scala2/moments/meanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
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
3siamese_1/scala2/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_1/scala2/moments/varianceMean*siamese_1/scala2/moments/SquaredDifference3siamese_1/scala2/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_1/scala2/moments/SqueezeSqueezesiamese_1/scala2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala2/moments/Squeeze_1Squeeze!siamese_1/scala2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_1/scala2/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9
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
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_1/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_1/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
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
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
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
siamese_1/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
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
 siamese_1/scala2/batchnorm/mul_2Mulsiamese_1/scala2/cond/Mergesiamese_1/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_1/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_1/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
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
siamese_1/scala3/AddAddsiamese_1/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
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
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_1/scala3/moments/Squeeze_1Squeeze!siamese_1/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_1/scala3/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0
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
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_1/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
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
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_1/scala3/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Tsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_1/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
c
siamese_1/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala3/cond/switch_tIdentitysiamese_1/scala3/cond/Switch:1*
_output_shapes
: *
T0

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
T0*
N*
_output_shapes
	:�: 
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
 siamese_1/scala3/batchnorm/RsqrtRsqrtsiamese_1/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_1/scala3/batchnorm/mulMul siamese_1/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
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
siamese_1/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_1/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
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
!siamese_1/scala4/moments/varianceMean*siamese_1/scala4/moments/SquaredDifference3siamese_1/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala4/moments/SqueezeSqueezesiamese_1/scala4/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_1/scala4/moments/Squeeze_1Squeeze!siamese_1/scala4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
ksiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
Nsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
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
siamese_1/scala4/cond/switch_tIdentitysiamese_1/scala4/cond/Switch:1*
_output_shapes
: *
T0

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
siamese_1/scala4/cond/Switch_1Switch siamese_1/scala4/moments/Squeezesiamese_1/scala4/cond/pred_id*3
_class)
'%loc:@siamese_1/scala4/moments/Squeeze*"
_output_shapes
:�:�*
T0
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
siamese_1/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_1/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
 siamese_1/scala4/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese_1/scala4/batchnorm/addAddsiamese_1/scala4/cond/Merge_1 siamese_1/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala4/batchnorm/RsqrtRsqrtsiamese_1/scala4/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese_1/scala5/splitSplit siamese_1/scala5/split/split_dimsiamese_1/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
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
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*
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
siamese_1/scala5/Conv2D_1Conv2Dsiamese_1/scala5/split:1siamese_1/scala5/split_1:1*
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
siamese_1/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/concatConcatV2siamese_1/scala5/Conv2Dsiamese_1/scala5/Conv2D_1siamese_1/scala5/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_1/scala5/AddAddsiamese_1/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
n
Placeholder_4Placeholder*
shape:*
dtype0*&
_output_shapes
:
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
save_1/SaveV2/tensor_namesConst*
_output_shapes
:,*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0
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
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:,*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights
�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*:
dtypes0
.2,*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::
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
save_1/Assign_4Assignsiamese/scala1/bn/moving_meansave_1/RestoreV2:4*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
save_1/Assign_6Assignsiamese/scala1/conv/biasessave_1/RestoreV2:6*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(
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
save_1/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave_1/RestoreV2:10*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
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
save_1/Assign_12Assignsiamese/scala2/bn/betasave_1/RestoreV2:12*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save_1/Assign_14Assignsiamese/scala2/bn/moving_meansave_1/RestoreV2:14*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
save_1/Assign_17Assignsiamese/scala2/conv/weightssave_1/RestoreV2:17*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(
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
save_1/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave_1/RestoreV2:21*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
save_1/Assign_25Assign!siamese/scala3/bn/moving_variancesave_1/RestoreV2:25*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave_1/RestoreV2:28*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
save_1/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave_1/RestoreV2:31*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_32Assignsiamese/scala4/bn/betasave_1/RestoreV2:32*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta
�
save_1/Assign_33Assignsiamese/scala4/bn/gammasave_1/RestoreV2:33*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_34Assignsiamese/scala4/bn/moving_meansave_1/RestoreV2:34*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_35Assign!siamese/scala4/bn/moving_variancesave_1/RestoreV2:35*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_37Assignsiamese/scala4/conv/weightssave_1/RestoreV2:37*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
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
save_1/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave_1/RestoreV2:39*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
siamese_2/scala1/moments/meanMeansiamese_2/scala1/Add/siamese_2/scala1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
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
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_2/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
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
Csiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
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
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_2/scala1/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_2/scala1/moments/Squeeze_1*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
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
siamese_2/scala1/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

k
siamese_2/scala1/cond/switch_tIdentitysiamese_2/scala1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala1/cond/switch_fIdentitysiamese_2/scala1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese_2/scala1/cond/pred_idIdentityis_training_2*
_output_shapes
: *
T0

�
siamese_2/scala1/cond/Switch_1Switch siamese_2/scala1/moments/Squeezesiamese_2/scala1/cond/pred_id*3
_class)
'%loc:@siamese_2/scala1/moments/Squeeze* 
_output_shapes
:`:`*
T0
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
N*
_output_shapes

:`: *
T0
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
siamese_2/scala1/batchnorm/mulMul siamese_2/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
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
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*
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
siamese_2/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
!siamese_2/scala2/moments/varianceMean*siamese_2/scala2/moments/SquaredDifference3siamese_2/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_2/scala2/moments/SqueezeSqueezesiamese_2/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
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
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
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
 siamese_2/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_2/scala2/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0
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
Nsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
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
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
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
"siamese_2/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
g
siamese_2/scala2/cond/SwitchSwitchis_training_2is_training_2*
_output_shapes
: : *
T0

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
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_2/scala2/cond/Merge_1Mergesiamese_2/scala2/cond/Switch_4 siamese_2/scala2/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
e
 siamese_2/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
siamese_2/scala2/ReluRelu siamese_2/scala2/batchnorm/add_1*
T0*'
_output_shapes
:�
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�
�
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
�
/siamese_2/scala3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
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
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
3siamese_2/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala3/moments/varianceMean*siamese_2/scala3/moments/SquaredDifference3siamese_2/scala3/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_2/scala3/moments/SqueezeSqueezesiamese_2/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
siamese_2/scala3/cond/switch_fIdentitysiamese_2/scala3/cond/Switch*
T0
*
_output_shapes
: 
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
siamese_2/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
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
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*
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
siamese_2/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
*siamese_2/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_2/scala4/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
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
Nsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
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
Isiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
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
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_2/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
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
siamese_2/scala4/cond/switch_fIdentitysiamese_2/scala4/cond/Switch*
T0
*
_output_shapes
: 
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
siamese_2/scala4/batchnorm/addAddsiamese_2/scala4/cond/Merge_1 siamese_2/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
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
siamese_2/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_2/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
�
ConstConst*
dtype0*'
_output_shapes
:�*��
value��B���"���+������a=h�̼}�<8�<�P�/�zz=���� @ʸ�Q�<<1���y��T+=0���8%��ڲ;,����Vq<�a���!������O=���F�<�2�H<0B����;*-c�0ٻ��˼�/<(�<1�<m75=��%���<Ϯ!=�ۊ<�c�<F,�������<Yr=��2=	q��K���i<'�W=<�<��=�A&=�<N=���<�u�; �0<��,<V= =��1;E�)=hD���74=ED7=�g�<Z�&���
�!/=��P;�
m=��,� � ;�NV�@׻P*<0�<< ~j<�E=L��< �r<���:�*���;<=X/�<@*�;�����/=��=<���<`�<tb�<����'M=$:��U�<������;�!u<� �<�d�<,R�ք%=��ȼ��B=p�y<\��<�=t`(���9�p�!�@��:�_��t�<�{������g�<����"������ ��p����z=`�w��0D=@S��h#R< ��:��i��8ܻ����{�;��b=���&�.�� ����0��<�+���L��D4��`�3������B=X����^�3��l,��$-�=`S��=�<��O���?��b
< �깼Z�<0�=�
�TgX���⻘8�p�����M���0�<`7T���� m#;`�v�~�?�������^��G=�=N����?�$����0�:��2(`��a�X�z�`���r�-�@�`�@Cf��[H=�a��锽l�� �;pю�J�/=�//=���<��ɼ�落 �':q����(�,����p1�<�'<�h����=�˥<p��$�+��<0�k�8��<��>�<d�<rs����d�����f� ���`���e<����S\��T����;�f޼ ��3r�\�Y����x,!�̻0�el<=LT�����:�����e���(��T�L�\Oz=�	�<����5�<ڞ8��ʑ� s�9 �#�������߼� w�z���0/F��Zܼ�Rؼn�|� Xp�@@Q�]\x=�j��r���$=���;�F!=hC�� �Y��ˋ� ��9Rkp�ű&=$R�<p�<��<>t��K�T(�< �D����@�C�Č���뫻P�<P�ƻ�.x� Հ��F�~\���ռ��ļ@ź���DU�=��!������<��������ଽ T&;t�f=�7�<@�=(-t<���;p�мK༒�� �ĺ0�< y�;�"O�@!ۼ���==
���uV� <&+t=��ռ-=����3W�|G��q�<����/�<진�0<߼�u=(�н���<D<=�0�h�H�"��0L< ��P�;}�=$OL�0�n<��Ѽ𼜻q�{= ]��ڽ-= =l�<�5��@X8� �;�[��.F� .�� �̼�"�L����s/<�m��hq#<𖶻��=DK����!�*���"�t��EK�a9����ӻ�]����;X�� &b��Ͻ�T��<T.�� +��1@��0p�LC�<@�};@��: �q���	=ts㼘�Ǽ���;�@׽�e��PV������C�<H�s�ڽE���(}�Lkۼ�+ǻ�ڦ�z8r�G�v��FW�= �(�1Ah= 8$���(�x�8<o+���-=F�J=�z���c�rHi��=���8Y�X�?=p'� *w<~��>)���\�	W��&w�`o���P�J= u;��ռ4en�^��s/����Խ�Q��˙�,"�~�D$�<��� �f;��Z���a�@h��(��<�^�0��<˙=$/G=j-��3�������m��=\��P�� Um;㯼�X�C��=�<�����o�љC=��λY`=X�ѽ@\����i�4�da��|H�������Y����<�B���*��@˽Z,=x+㼰	�<� A��ެ��Mg< 셽���L��<�r������k����>��h���Խ|F=��B�{��֑��e1�肎���x����=`�*�$��\':07<�=��!�@�E<Av���L伀hd<@��;�\��M�D�=O�@=�]�<�ƻP&��SӼ�G�R�� ��9������<=�sG<��K��c��4Yf= &:��ټ`�m��2B���2����<� !<�n�<�������<6r�(&μp��P���
�N��s=�	�8
� .I<Hlʼ�Q���	u��P��.=P�=�B`����<x��<�[��(N���G��ʬ���FE��PQ���x�� >� �==��
��N�L3�=�6r�p�Ļ?��� ���4[��(�8��Wf�('��0���@d���h=�����d�<r�==@�|����x�Ƽ b9�%E�x<��< j{��;~�@��;�(����<�%=��=�k(<H$)=θ���== Ph;{Ī� W1��������7��aμ�봼4W+�hYt�0����L�;>���d�<�x�V^x�@�;������|�:�<�=$i �ZdG=����"�����:�L��ҍ�/e��	2�!L=0��Ȇj<������=�D=$���@c��� �C�����5;�����Z;�3�Wv�t��l�<�������V���ѽ��Ѽ�橼>�	�k��=L�<fL�=Q�z=p��`�<N���	a=�7�=Lj���h=W����?��"���=h��<\t�<����F�ؽ`[�g�ý�����+�P:��@��<����S�<Ț>���𝑼�qR�����0H?��ō����G��;�=�iм.$�vf[��N��쎽�AS=�E\��ƍ�2��=�U�=�j@�z�Y���<�̻�+�=@7���i�<8����TB�C쟽}�=S�<�ý^<��̔�=��<g�=���8��r�Q���5<�ş;�O�;r�P|ļs�=iC��Ĕ�|ý�U�= ��9��5=�Ş<�/}��t=�VŽ��k�`I);�㼽� �L_t� �^;�����ʝ��< ̅;r�0�dM�8�6� ���҂������q�=ҿ=b��u1=�Ӈ=�l�;�-ԼC�M=󡒽��;v)%=�����l�L>��xl�Ǹ=ت=XV/<4�;��� ��0���Ἅo��"�F=~��K����KP�=��:&5J�̕�<z���L��pQ��ҹ<�A�=��/�`��<��� ��p;� �"��я��l�<Ȧ��t�-�X�<�<¼���Jk$�������<j�=�*i��&�;&�= ���B0=9���5�lo1=���@p< E�tab�t��< �ý��<�3�:�5�=��O�������ԓ鼄H��	��/fŽTNܼ`�V�h�H�8/�<�����=*� =��^�I퍽�Ѽ�a�;�Պ���R�8�ͼ(�� 3'�x$=���<j�8��Gf�X�G=hsb<�<�TP��o=�t;��ֽ��"=H���%�Z�/���]�Ԅ.� ����U����xo�:�ݽ.��I�Į޼���;��۽#.V=��=��=�
��hS�=vs@���$�����F3���L=��{��)�LoE=8tb� $U< xܼ~�=@!�=��8�~����N���[����2�.�䁛�p�ϼ��a�������E�=қ�^���H <���<�B��ۊ=�<�=�]=��=���|��<k���lul=6=`�J�M��=�!���{?<���3L=L�=@��l�����l�β�3v���`V���Pxv����.�9Y)=�}ܼ�k)���;���;�i��,g�`	�;{�����w��=�䋼��� }S�>]p= 59�ٕ=�L�PI�����=:7Z=r�� �)�om0=`!;`r�=~�=̕�<�9� �E�zM�.��=��=�'��n)���ng=���<)m=��~�,(�������=�l�<��k��ݼ���8I�<ع񼨜<0%��у=��=�	�<���=�����=�V�ӣ>=`u�;�S^��R��8Q�� #L���� �t;�])�g����g=H!����.� ��<`��;��'�pI�>�=� =�G�,�=7�=L;�<�Nc�.�;=��L�0󤻹a$=d�=�x�����< 
U��n>,��<x�üT�����$��{�p#�ޓ�ǎ��Z7T=�ԇ��%��ጼ��Z=�-���A���=<���0��?����;N�=��$���ԻPf�<<w��ag�@���!��� �u�`�|�<���a;=���җ���9@��֎����;�ۭ=���A���f=�<]=u�߽8_?�D�=љ����u=P�;x"���;�䔽���;�k<
6n=�D�@M�; � �S����n��4�����K�Z���n�	/=�m�<t#0=�&
= 4(�7�������:�<�{ͽ��F����� p�� �\;�9=��=i���P!b�7�W=wi=�)躀%m�==��t�{׽%�<5��Xnu�.�	��ɭ�d�%�����������L�����v� <��W���e^�`<�< }����=��=��=�u���=̕ۼ�ܞ<`���^�J���=c�����@x%=��+�ւX=и��aZ�=g�=�:&�8����F۽�i~��:(�%���6?��{�VM"���ν����-=�8'����*u=h�j<�Y��W�=��=��=#2�=�<�"�<�S��Rz$=����.;�V�=�����=���Nd.=�H=��2���ȼ�v�6��zXk�,-�P���ې�d�μL:��uei=p�������<; .�9���Ы^��<��~*�o)���6�=�Į;&Ec���v��]=�0ɻ��X=:� ���9��h�= M�<����Ǆ;T��<׳-=ø�=�R=3�<��	��g��򺽔�==wkQ=�ǜ�s}����<nE�=\��<
����b��mμ@��<x�f<l詼ᑁ�T�G� ������@U�< �:�\= ��<P�<@= >�Ⓕ��c<P�T�H��=  ���ꇽ�J����< h�8A �ȎG<У;�\�Ҽ�w�=�P�<xТ���;��[;��j<����>-̐=�w6��U�;��= :�99]P=̶7=��w����$��<
�C=ҫ��P!3<4Ġ��&�=pU�;dY	� �¼�3��ۻ0:ټ��r<v����}<z�6�0s�0ҫ�u��\������q=h���|K�<��ż���B�-=�� ����]=̠_��`��� ļ����nO�p�����H��<�a;�?R�����p� 8���b�;<���Dp���=0uϻ�U�=R&b�xj�ˠ�=�䍽Or=أ�<��8�@g����߼M�<������<�6Ѽ(v7<���<qzƽNI���[ϼ��Ľp溼̄� �7<Pu<v�=���<�V<��<���� gT�0(���bf��N����R��+?��_���-�<�^a=�
��"�d=�2=���֍=p<�,�?;��p��;$鼐{�;HA<������������M��?��#�<����?$;������Q;0S����\-�=�[���x:0̾��/�;@��: };��g�, �����=������䭾<X)��#=�����_�<�]<�a��D�5�d�1�,#���*<b�� �G;h�� �����y��OA��Vl<X~�8��O��d�-=����=���<�y�;��o<��-�x]�<��� �G�T�@3��0D9<\Pܼ��`^d�\3��h�m�TM3�������<|6�����P!C<�[���� =�>$��GƼ��;�X���<Բ�xiv��ɍ��u���`�<�м 4���E=��� u���r�A{==���<@M�:؛]���;�H����\��|�=�鎼Pw߻(6м �e:����\1�x;.<�Dw�`h��Tg�<H?��1�HU��$ݝ<@A��;9<��~D*�<������W�b^'��a6�Dg��X
��IP<Puu<��;�������@o=(W��ݼ�
��.�<�.<�u;�����*Y:���8� ��=@+�<�ͻ<�g�<��=��E:@�»���<�� 8{�`�%<�誼&�0G�`�0<0P��<4��<XO`�'D=�j2��ѐ=�y��Vt=����(�-� :Pe߻ ���(,�<09�;(	�<���<]���KC=Hh^<�D;Є<��G�L�n�)����=�*���;�y6��p�;�0�<�R�껁=`�z��u<�5<PX<�+�<�d{�
=@N}=`'��`�; 奼@�$����<��r< Tx=��R�����$�L=\5=bDy=�>�=(˳<_:�=<�a=4V�<��=�)��eC�=@k};@I= �G�w�r=�`�=��X=�5� �i��Lk;�B`��в<xZ�<�n�@O;<�����!=� �;�ߟ<��=����@��:��<�=��yZ=Nz�=��K���;��O�=�d��\�=�/�F=�9���=����^-= Ɓ�@@�\��<��3=5&3= ��RR=8���֫=��(=D��<zu�=�\/��w
�8��<���; �¼ǖO=��=� =@�r��^м��$=��=��^�&���=PH.<�L=��;�v�<�M�<h�<�&���\��P��P!=�����A��~�P�T8��D*�< yp��i_�l��< <�L������< ��:v53��f<����:h��<@#�<��������g�l8�<@p8�T�<��
<�X"���ټ�9��� <ܻ�4��� ���8�P�»�h�� S�H�:�	���+'=��=@
~�>|K��+A�\=?���?�X�x����; B?;8<��s<X���_��,W�/(= �v�$�,�(�y���<�e��P�<�6�:�Km��0�;v� ���k佽�~n�H#��̩E����� 1����=��'=�c.<(L�p�� �9�_��XG�<�����~���?�Z���F#�pnۻ@[�<�T�$��<����!6�R�#� M��㎼�������Dk^�����19�O,�jX=8<p��;Tʣ�&PI��^<����ļ_�<=�e6�狁���=XH���V����<�	��v����6T%���`�H������@:�����@%�;���</*�=0�t�$Ñ<�` =�`D��c�Шn��^$���(<�
="$�5�N=�<�ܶ=�,= 2���~��#��м��#�x��<p��P*��sx<�`V�y��L��<�q�;sn<�q���Pü0`�;@�:ԃ�= xo;`���[L>=�aż����B�q����<�M=Xl>�z�v=�	<`n�;P�L<�E��xL<l?�< X;���=`<\@¼�{7���==��񻊒L�0�;���=X�_��z=�7(��Wͼ�2»�&�<��ü�\E�]Z==���Z=2��<�5=��=�ϼ�ߛ=�NO� �<0Ar<H}�<�ݳ=�#=�����i-��`�<�A;=R�7=`<�� v=`{�;��� ��H��<�+���M7� ��;P7\<�;�5g;�VD=h���3�=�")=.B=��=�z���Z�����P���/R� �;�:�=�N=����t-�<Z[O���=��Ż
�.���):'��M�;Tl�<8��<`�D���;F�8=�����g��7Q<nH޽�jD�\̼�ƃ�`��<�} �W���b�i�`:����lƗ��]Y�jv6�Zor�������=@�����G=� �<8b�lu�<�����'=O�=�|��h�C<�������T��yt= <���2�;L䱼\����Ia�� f��7�t�h�&�+l?=����x-�ܮe���Ͻ�v����:��xG�ؾ�� �����N��tɽPś<j�o�������T��g���Ž�I�<�����~<�<�=�b�<���:�T��qc��:Q���<0�߼�����N��J�:4�J��=��[��<��V[���V=XμD�}=����kW�\�׼.�'�~(�0��;ȵ��𰶼]-'=�&�\�'m��!'G=�*#���<j�^�LK���Rn<�+y��-�P�v<�� ��-��ɣ�T���Ԙ.����,ɽ��=���j7�����߼�7��`
�py�;2�<��3�f�$���׼��0��^���I����<"�T�x;O=��x=���<�����@�<Э<�<`�ϻ��7��N��(d[�P4<�׼�l�<8d�-�>�ҿ<�>���lY�`��x;*<���;@�:8�$��Ō��J;��7�� =(<l��7=���<t�I������4��@��;3=��#��Ǔ���r=�OO�8��6d���=��;<����BE;�r�<` u<$��<��<�	��0��@
��= 흺XL#��v�ǡO=|t�VfN��nﺳS>~f���'=�D�� �ɻ�z@<��\��q�m燽��=�P��|�>=�ֽ��\=2�<�zh���=�����,�<�V���@�<,��< �;��N���d<pU<���< ��:�����==`�<����&=߆<�C�� .:�M3<�[<�RT�@��H��<�w��?�=�77=f� gμ ����a��zAf�P2��Z'i��k=���=ETz=�����=J�;�/U�=D!�Xd�`�W������ٛ���<|��<@o޺ ��_��=��j=��C�H!��8��H�*������F&;���HȽ���������X��G?������弈��X�"��C�=#�<�"e=_�=�Ọ��<��c�c=$�V=L?v��{l=�ǥ�wۼ4�ϼd�=��<p��;(�G��vҽ@���o����(5)��\�� V=2�B��ۆ��`;�rU�����;��<s�������@;@�;�����=F�F�j#��(��c��0ݔ�F�h=bQ�@c�����=��e=�jZ���˼8�]<L=���Ϭ=pN�x��<Ģ̼��&�6�,��/�=P9/<��ý��s����=�Ơ:�آ=!s�hʡ�jr��T�Hb��8ۛ<�6F��l7;͆0=�#� F�񪛽��= ����
=�ܚ�Pk\���]=�>���>��@�ӻ|�Z��A���u���N<H)���s���Ƚ8<�����@,�8r��:�����y�04�����=�÷<U漘�-<��'=�<�g)���C=�]�b��=���= ����'k�(�<8�.�N�P=�-=PȈ���H����2� ��������;����5�=b�� @Q��:P�,B�<�4�<<���׃=�f��X�~��;6�X�#��=F��W(=L",=�H���X$�p7ϻ�����R���a�j����S�=N�H�صt�Zy1��,<�*���<�,F��-|�LD=�=Q�=u���lҼ�n=�����;0'���z�@I�;PM���輐��<��=�.��q�;.���* �@�����[�� ν
�R���t=��y���w< �2:<�=آ9<�.
<4L=�Ə��>=���0�H<�e�l�!��>�� �=���< M� һ��R�+�e=����7'<���<\�<*}�0;|=���<��M<H+<��࿗;0%���h�<H�w<��mV�����zx�V0��	B��ƃ�G�=���=}�s=H���k%�=�����ې=�\=�~O`��QZ= ��P��; �ź@�; �U�,�&�.�W=�v=�\
�^8����V�غ-�|&�|��� �p������G�|���������<�D��P�5���*��1=����?,=*��=�+=4��=�����#=/ˇ�^�v= ��<�� ����=*�=��[ ��+��|��<Px�<�r����1�d�A�xǼ-���8�X<�N�H�<�Bi��S9�dU�<�#�,�
��<�<�B(<�j:�Pp��X��<8�o<ε7�Є�<�>��"���A�<}@=@ϻ* �=J�C�@��:��^=��<Ж@���<��<=��ͼt�=�G= ų�P��`&'�Зw<�c�={�=�a�� ���P= `��Q�m=�^S��>�1l���9V< ���`�o;0I�;��
�v =���(�<`=<S�l=/<=`�Y��B�=(���H6�<���3� =�'�;�߲� z�;��� Ť�4*����[<������H6-=BO � b<O24=P�~<����;� �=��k<�"�E^<y�=$^�<.=�R/=n�V�0� =U�0=�z=H[P��>=H�c��׮=	�Q=�u��RW'���F�����`V�;s���Ez�=���@����Ἰvm<`�;�Uʼ�_=j\Ƚ�3ڼ@(+�`R0;���=0H���k�<�s=$�k�p�Z���F�90��,����D�ꟼ�D�=�+��&ּ��)�M�h�X��=�&�]��CQ=O>5=���=#���0P���="��]0"=xÉ<��B���� ۼ �1���)=�Q\=��"�U�<� �;�Ï�L��Ąɼ�������4<�W<�=�8N=��<(9_<*�=�}�2�m�!}D=��^�?껺#&��',��}��6�=j)�=X�v�<�<�v=0���<F�<�С<$P�<2�e��>D=�D<-Z<���;$Q^�@_��sּl���ܵ���c��P��XX���}�������;��G�=_�t=��(=9��岘= ��;��3=$�żF,�3�=��޺0��<@��:4Ȓ�W<X�b��
Z=��V=zj��ج���Ľ:5F�j]&��`���W�����JG�	*��0�X�L�<����κ��r�<�؆<�ң�>�U=n?�=?��=��=P�}��1=7�ɽ�A=9��@!}�NQ�=����@��;�;Ƽ<ڸ<�L=(L�f�"���Z���i��d�DH�����"������9�==\��=)��<�E;.�U���� �l�h�S������=�|5�GE���lἀ;�<L�ؼ��=6,�`���׉=�3�+���`.<��=�T<�z=VnE=PG�;X_�l���u���=7�h=y���.����=ʅ='�0=����P��`� ��;PX�;|����$��]���D;�ȼ��< �4;"�O=��<��J<���=�N��(�?<�j4�=|}=�^R:,�V�`Fn�����@���!��� :<��O�Z+,�RĽ= �ջ e9�X< 6�; �};���;��>�x=
|�PpR���L=@��;h�f<�4C=�x�����,x�<�nd=�ͼ���<��X� mj=(��<��H�࿲��yؼ���<�Z�$`>=x�q��D=�PE�t}=,厼<|���S��02ۻ1)=vM�����< )ú�z<|�<�����<t{=�� �;�h����t��ȓ������Pf�;�G=�	��؆Y�L
�`����;��<�����((���=p��;���=�)Ǽ�[�;�y=\���\��<�T=��޼P�� �W;��n<��m�P��<4����<�'=1����W�ä��$����e��l��;�<��<���=�[�<l݀<�=�	�D�����;gl�@׼
�R��g��$	��F0=*@'=�";��O;�=-�= �ʹ��)=H�"��wo<$���A =���<��<P��<��+�H�P�p�;H����F�� x�:(�o< >�:`�f�`�[��7R����;}�=��<E��`17;`�;La�<��<�传+;(��=`J����; :|Ԧ��8<,W%�P,�����|�B�P;�Hö�$����d��@S��w;���F�`�m<dqݼ@�������R1�[o)=�M���E���R^= R?�8�=<`m-��h=ZK5�p>��*z4�`4м0N<ԣ�m~�� O���o��8����F�P�Լ�>�;�d���y=�`̿<D�"��LK=���N��������H;��,w���o/<��|����<�"κ�B4�p�S<�J��7�hQ<0�<�(�:$�<ltܼ�R� 5>�ȰC�د����=�	T��41���d��T<�4�4���C<�݄< ^<d�=���H�Hl� �� ��9��ػ!��#e�,<�N��o�ࡊ��3P�$h�����T�<��<�8��@6�:���h=��B�X�Ҽ dj�|��<�'<�U�< �2��B�l	��� �PB�<��`<�Sp<��V<@D�;��; �����<�}K�f_<�5��_)��5��0���.��r;x�ϼ$}�<�d��5�`=�4��&�=�kM���=P㪼lﹼ0<����@B�; ��;�< �ҹ�Cf:������=��	=�A#=���<�į��/��@����p=4�f�3�<�T4���@�T���Z�m=�L�<�
�=��@��g�<�˛;d証h�n��5�<L=�@�=��Z� ����ۻ��&;��d;8!+���r=����`8s<d=�g<݀=N(�=��t<+o�=�3=��<Hg~<�ef����=p��;�3<�ӫ���= ��=�x�=2�z�( H�����NO�|��<��:<`�i�����@��: �H=Н�;@��<���<=��To���=�A�<���<J+�=�#x�J�	��$�=�}�A��=�;��L�<_L��(j%<*���=L��<���;X�N<V�W=>�R=��y��W=��<l=��==w
=\�>xR�� I��EJa=����n�J��*o=�s�= S�<��'�$R���=�<`�>
H���)���: �P:���<�.'<�"=�O=8N�<`^;�
;��=�-=�7���t�@*�* R���6��Q��� Ѽ�⼐ۻ��;P6<@I�ht6<nj�n�,�굁=�� <�J=	�< ^����	=@�ڻĝ�<h������@�t<`m:�`��<��;�!�d�<�Fc����7ۼP�� �)��殼���`�<*4E=p�d<�Ƽ�2������@�<p<��R� �f�`%;� �rF� sJ; 葸�!˼jBf�@�c��c��O�=P�{��hV=�!= ��:L�޼V�i�0r�;F� �+9 �;�ϲ��܌����<^`�A�I= �:@O��ؼ�M�;0����#�<&뉽��W;�b[�$����@��O�;@`���gN�4��<�D1�8^��:}�#?<=�e��p�< ��������0��� t���e=�8���Ļ���vo�hC��� �8WH���=�h=�dڼ���<j��XuM��p5;=�3=�<�{<<d��>{Q���� ���vV;��ۻ���=�
=�d�=̇L��	=(ӣ<�� ��d��9�<�L���<Y=�.�`�=�߼<}r>C^a=XV=�yB<Ż���X０?�C#<��a�8�$�P��;�k��*����=Y�R=	�)=0��;��(� %!:`V��q==@�<�Q��� =ܐм��м��ȼ�7= <=����X�=\��<�%L<�@�<�+��L=I�x=`+T;�0�=0�#��F!��&�:�+</="¼��w�ߢ=��<4�z=@���^tE�X�Y<���;�^�������Yy=�U�l�<P�����b=.j=ԓ���6�=�wf���a<�A=��=^��=��j=�� ��o)���<P�;���=�X���V�<��� ���
�.=�咽�Ѽ 1^;��=�<L,�<��Y=@��B�=�I�=~_�=dq= ے�г_���<tK��f�M� �&<"��=MeS=j������<ʀX����=����������
��;M�h��<��Y=��< ��<�K6=�1^�O�=�k#=��
�:d�(k��ұ��*� �9��l�Oܯ�P1�� sU��:��|�'�h�{<n�Q��_\��$�=�p���=�j"=�Rv�P|�<�����=�&�;=&���=~\���<��~��T5=��8=x�j<X�b������<,'ռ��E��򒻠��)��= �c<�ܼ�A"�lT̽Pv�;X�#<1���Xx��rk��ȁ�rн9�b= Q��yf��8��Lx���S���=@zB��-=�[�=��<�,�+��� �9 4���-=�K7��8<���: �D;���=����i���&y=�~=,�.=�{�����<�]�<�\�ȝ���H�<>�D�����4�=L������S�c��=��/��.0= 2z:,壼0��<@l�8�=�x4=Ғ��X:J�������P�6��>���ʽ^�<�7= ����@�~�����0i$����<���=��)=�"��j�=���<����^i=��A;T�[/�=���=��3=\%��D�E=@~V��ja<P+`�dҼ����@ɉ:~�=|oռ�s�<�L3:�V:>H=�zn<XZ���h�� �:\�< E<�օ�t�S� �⼅���~»�2�<��|=P�=���p�L��sq�.����=�D<`�{�M�=&BC�t�>���缹ԉ=�Ԍ;�]�(=�#�<@l�;�(=��A���<���;��\<8��=��S�X�6���_��L�<@��:����~��n�>$�6�- {=B�v��E���=`�N��<J����f7�=V��HrJ<�{ݽ��=�Lp<��׼���=���7{1=��V<�A=Tڎ<E{$=�ӷ�@۔;P�U<�Y�:cM=
�����=�֒�h�<�o�<ȷ�<����`�;���<ύ<=��G��i�;}$5=�k�"?�=���=@�;�P<�x;jL���%��4쭼�>���j=�/�=̎�=*3��N��=f|U�Z�>�S��V�(�2���e �\4��P^�<�\;=�<hf뼁��=�<�2= �M;n���[�@�%;�@��x�߼,�J���Y�已����� b�;�g�`!8�D�<XKE��?*�ӵ�=K�;~\�=� �=h�<���<�z��f5=pЛ<Ԅ��,�K=
?�<x�<xg�#�= Ʉ=��<��ջ�����L6<Fv�R�	�`t5�,��S�f=���;�;<�-���g����< ��<V9���M�H�@`U�2��Y�=��ĺV�H�v�i��D�����\=��5��=��=��9=���z���Jk<P9<��=�!Y<�q<`d�  n������[�=�8�;���L�	�^�;=� c=��Z=0u˽�,�< E��_ ���O���<.I.������<���:�SK������t�= �y�D=\�=T���-=2������;�=�,��x��ʞ�h�<��{��YQ�ޯ��`��<�@=p ����t;����@��h��xZ�<*Z�=�C>=8Y+�@+�A=hu<R�=�߹<��\��=M��=�$u<���	(]=d?Z��T=�$<�;��џ��<�����O��s&� ��;�("�>�2>�3�����<��0���?�L3�<0u�<v� =��� �c�x�{��*3��iW= �O:��4=���=hs��濼dT��ǐ��ho�<�4�8'C�6��=BJ8���ռ�Rۼ�`=�
ļ��v�C�@�:�f�<x�]=8�=�(%�0J�<�W=���< �%�x#㼄91��oۼ0��;\ȼ��<��= �b�h =
�=���1�̥�<h1��O���_���e�=�h������N��)= ^� M�<���=ŝ����w=�O�`�=V��@�1�O۽fSb=�m<��þ/=��z���Q=*��	=ds�<t�<ދ-�$��=6=M�#=,�<8Z���#=�8ԼK��=�'�=�����m��p���Z��|ּ(�D��08����=Q�=���=�S4�O�>�卼_r>�s��J �Za?=�����<��m���=@ď�x�K� -X=��<�L�<� �L������8bo��$E���'�Եm�@�����4�PM޼�m=puo����y=�L�<�>]�Jæ=(�=���=���=���;��I= �+�W=�	~�L��<(�r=T輍k#= �� &=���=����=����μ@kɻn�(�@y�:譾��E�;(�g<Y<�[`=H张d𼈤�<p��<��F�`<� P�7�
w;��:�%�n=3@<
�P*ݻrk =������=P���=�=�=nJ;�0:�;^+%=p&�;!��=��m=����V+����Ė���=X�=�l��YF���=��=؞+=�;'��~;:�4�T��<l٫<,W�<jX�PGV��RY<�C� )�;`�8=b?�=9�<�)i<���=?<��<��	��2C=�p*=To}�@�Q; �x; ��z����c�����r3<X�=�Zb�A�<4��<쿡��L�;���<'X�=�@=����@��;X�U=�M�<��=���<��G�C�=�{�<��M=���#h=��Q��a=[�1=8y��ߎ�\^�H�%��ղ:H�V<����	>\T��p��<�܈�����w�x5<,oF=P���E(�p'�4���~uk=sN<T^�<�/�=C,�t*%��,��.��0ࡼ8/#����<8Y>Dv�`�P�F���5&=p����秺�N�0�]���)=~[u=�?�=�o��tM=���=�Iz�\��<�<�/��x��|d�< Һ�&=cC=�����=`U}�R�c��_�<�"м�Z������P=<�<��;��>=���<p�1<�g=f�f=Od���)0=H(���< �;��bK�Ś��?�v=
�3=�����=���=�>=8Zx��Z=�G�<HKO<Zx ��dv=`�<0��<�5Z<H� ��A=����(Ǿ<��=��T���< o������������;,�Ӽ,�>��=�C�<�u���=@}��^�=�A�h"�"=�= e��[l= �i:�ۆ<Xko��y�>��=�V=��<*P����?ٱ��<�e-������V����;���(i��4	=�S3��½���=pZ�;gH��Wn�=�֘={f�=��>({d<�@=K����_H=����u;q�=��Ľ�=�eؼ홊=���=�6<N �,���0��;J��|B��DC��)� �b�h^�<1�=p/�O� �<ѳ<�)���+��I����Qн��>̈<=�Cý����@�7;xA��q��=p���D��<���=f�<����"ļ�N=�X=��=w�y=dc�<0U1�@҈�M@���P�=~i=���Ϗ�v�'=�a>��=z���
��S9;d��<f=�����u������Q��������:���<_�=�h�;�44=rq.>D��<t��<d���f��=l��<�j��n�Ra%=Pʦ;�0瘼^s��lp��+>|�<�⺈Ζ����J�= :�<<H>���=Z��z�j�= �)<��=W*=#W���	-��^�Z��=�����=$R$�Vi/=��<|�n�r#��@M�|��<T�м1�L= �b<�= d���kr= �лB�c����T9�<�s�<��@���X�u<�����<�=���<h�=$����*��0�
�1��ڲ����;'2_=D�=��(���<��ļ���<<�
��-(�Ō<P�⻔�=/~=��H=����=s=�i<= ���00�<t�=�孼�W���a6=��<��i:�Yٻhz/<t��<�[�<ut��:��$U��~w���#� �<��<��<͇=,R�<8u�<��k=~�=�&� �q<���<��<{�`SN<�哽��=c�=p��;T= tc����<@��PN�= p���8}<�q�;�8= g�|�<L�<���2�<x��<��y<8�I<�|�	4[=��q<�<x�X܇< 53��7�<��=�::L�μ��<x���l;Z�U=����Xç<��=D%��9<)= �9Ћ�;@����A	�𾇻@����<x�0$˽�뉽�,N�����()��$Q�� ��9��1�p
���0>��T�M��f��.J<�W��uL�=�?=�-c=�S=x�����!=�u!� ���"�j� -_�T��<��4��}x� l���GA<X=`���w����}���;�0(��̟�����0;D< ����T< �9�Q ����@��� �s�L�0sK�H0��(��.u���x=(D�<֯r�b�<�Rͼ�z�]�=:f<P��< �c=�F6��-����H�P�-��`�;����i�<xVP��v�;�V�<�҄�4��<�=LqT��9��@��:�݌=��s�0=`�l�ּ�y��m���n<�oK��n���s��&���d ��.�4V�<�j=�m��u�;��=ز�p�;4���l��<�=�{n�t頼�#$<�K�ڃ�Ԍ��0������<l,�=�c=XP<r���s��\"�<�F=�ח=HZ<�K��1���4=�m <�!�= `�8����
=�;��K�=��`��M=�˼RT����lM�<Q�7= v<���<�i�����<�v�<�=�D�=��=�==�Rܽ������0v=d�:���d=��׼�T��(�8��1�=���=`Ec=X�<��/=��컞��B�B}=?ii=�,�=�缠�4��9����<�#>;��?�`YS=H�/��($=hGN= J;<�.�=G�>�<8{�=8�<H�< ��;���x��=��<�Yѻ��0��%�=AB3=�=�^��l�μ^�5��I�S}=��< ���&�/���=��I=�z%=�=���:�͏�$f�<]�L=`:=܏���v�<X�����D�:��=�OL�P�=�H��$P�<�۾�� =ʼ�dQ�<��2=*%m=@��(��=6�7=�lz��=�=��<�.=UNI=���=�-�;��3:P?�=��.����$�c=HS=0�4��G)<��[�@pv�mz�=D߼  7��<м���<(/�<p?�=�E�=��<�vҼl�n�3�@=W\~=&��n��.���7x�@���������40��8�S� ��;�z�<�R��l̽<F��I;��~ͫ=h3��W= ��;�ª���<�*#=���;*�7���
<��@�G;m2C= /�;x�T�k�=�����.���컀2V:�m�;�['���;���<�M=y�H=��W�l�ȼܭ���m<@�; MG�Ўֻ���:��S�
�� �=<EyE=R��˯�����T^ԽL��<h�W<̭�=�m =���:�(2������<�<0к;���<0~,�P�;�/==��O�ԛ�< !�;�XW<�������=���X�6����<\��<�t�*<p�����鼬�j� X���Lu<xi ��Ҁ<�DM=���0�;�d=��< �9��K���<
ݓ=��� ���̔��@7¼(�� bH�D�<P��=pO�� �<܎����}����<.�==��<ؓ��Ib��&�0�ϻ�j{=����X<��p%=2_=V�=��>�H�
<�N�Pԭ�D���ٮ�='l8=\��<x�O=�G�CN+=q�U=X�=b��=���=�b9=]J����$��*< �Y�P�t��;=@��,e���G̽`l�=+��= ��<6.=��=� �0k�;�T�<k�J=��<�Q�=��<4�������l�=(u=�ru��h�=�N�<\m=p��<�R\��v�=���=��<�=���`�����<��7��+s=��<@����"=X�<��<4�<r�g�Щ <��g;xjN<�v�<�o�=8E��;��jY��N=P�=�r���v5=P�˼�=6'�=�:=��	=�?�<�v¼�����m<�_�����=(�� %;�.�� N�< |���<�A�8��<�I��_x=�u�;(Ʊ<��*=���<�Bp=�c�=+�=W�Z=0�< ���stM=��޼h=��`H+<�Ɓ=f�H=���� �һ�*���L�=�a'����;0������� H�<H��=�b)=0�<$*� H3��N=��=�����.%� 麼�2���8���M��V�������"���;����<^�%�`�Xe�=����@= f�;��漸��<X3<��<bV��Ć� �;Pq�+�+= Q: ��Ĥ=�0V<0����ۼx�I<� �:�����S�;@�;�v=��;="��м�I��4�< �� B ��
Ż ��:`蜻��=�H1r<���<�;�u���x׭�V)��T�<pj�<T�u=�X=@^�^�Yە����;`<e;�;P�w<p-�8P<�C�<R!e�W"= ��8H��Խͼ ��;$��< ��9����H�@<��=��̼0J�������E��(1#�  �;�R<�I�P�S��z=`���Tw<��<��d<�1Z<`�k��H#��u\=@o�@lٺ@q+��c����:�4t�����$¬<�1�= �����:��Tp� ��;&	=-�<4��<��Z������b�H��YAc=����� �G��='o=�n=\R��p��;�4{�P*N<@౽B�P=0�м�%);��=�ʼpH�<�f�<,�>�ݐ=05[=�s�<�����7⺅�G= �:�����TQ�H3��ǁ��mb��%z=�3�=#�;=h�V<���<����0<��K=�E=�P��4�=��;�l� Fu�J��=�:K<ڃ!��݃=u=�:j<0\<����� :=\M�=��<.bq=�����4�� U<@�;�T'= �[<�"|��p�=����_�<�g#�@����}#=���:�i)�F0l�6�=��
�����D���ׄ=��j=X=B����=��qS0=D��<	=��<�=գ������0r	�
�H6�=
Y���yi;~&0���.=�:=@�:_9��N<��<2R�=:���1<bf=��:��=�j�=wV=,��<�߄<�F�<ԝ<�¼��o�}�X=�n=���=l�f��=v��j�>e����U���*�`�t�[A�"p
=��[=�f�:�+� ~�;@|(=�w
=N*����6��y���ټ4�ϼ�I�� Vs9𻖻؄]< l_�`_<�q�;`?����[�I=I�<4F�<x��<��,/�<��<���<�케�;p&E<`)k;��<��;з���4=��<��EQ:��;��:0�%��%<(�0�(j�<<f=`��<����������%�< �f�PҊ;�r�;(~Z<���< t���N<;���;pI�������ؼ��{�T�=���B=���<����~~�&¼`�<Oj�l��<,6�<D�� �麀��<X�,����<�Iv<�W��Ԋ���} �W <^�;�{ݼ r9;,���d�׼8O#�q{<���_;� 5Ի�o�;���W�<�VQ=W�;P4ϻ\y�<��; A��|� ��;��E=H��@j�:0�P�8������d]���!���ǂ<"wR=Юϻ�j�;P�� -�������6=`�;����*¼d�ݼ : ��O�;���<P�(�<�¼&��=(4=�'�<tB��X�,<]��7�<��2���
;����F�p�;H^���`< ]�9��>8=�<_a=`V8����� ��:��a=���<�K���#ռD�Y��TG� (�,�=��<OrH= �[����;@c��������<�2�;���!E�=�K
<l�Y�����\c�=�鸼h#���b�<���;Œ<\R�<���]�;�з=�U=�X <L����(���!=�#�G=�=�|����=hK�܁�<*�R��ɼ��0=����Լ1������= b�����P,
�IZI=4O=p�%����=��L��(=��;f�=��<Do<.~	����;�c��X%�N~8=^&��hƦ<�k��F��=��J=�.��N	Z����<c�<�}K=�Ȩ:�Iм�2�<���<�P�=֪�=�Á��r�<p�?<�r�0�<`�����D����=Z�=�̉=�1S��_�=T�|>�*�� i�9��$=P �;`�һ��<�f-=lO%��M
����<@��:@>D<�������Ҟ;��ռP�H�<��ɼ@X�;�jw<��5��B=��e���;��<5w=����$<�<���=T��<R��=����GL@=Tt�<�Q�<t[���h<�L"=@���LP�<�ZQ<I�E%]=�JQ�@{�; -�<0�Lu���8
=�mɼo� =��0<T(�<�B�<�sJ�@k2����<ت@<@i����<|!�<U)=�>��ru<@e�;��.�=P��<�y$�ǅ{= �q��!=4¾<h㈼�j��Ģ<(��< �h8X(=be=��Pڋ����;X�3<t�=��D= r~��4� =:���<��`<���0m������&:�O.<���< �9�iI��5�`���m�<�ׂ=q=)�6=�0�g9�=H��<@�"<pZ����*=��=d���`��;ؤ�<p���\*�Л滰p$<Hl;��ֺ=�Xo:�T�<�2�<�LA<@.
��u=w
=��Y<P��� �V�,�<�c�<|��<�Tk<pPԼ��<�W���E]=,�\���<x���)= �廐w/�WP���`=�`���\��i�<`��;���=��<Fhr=�H�;cȊ��䄼:$:=��+=��@�D �0Q��\���H�\<XM=��l���=�<؍��@塽8ڸ�`7�< "�;Iw�=q�>hYV<P]�� 鶼	�A=(��ﯼ|��<�����^ =�3=�W���w��c�=�yj=����<P��H�4�\��B_�=��}=��C�`{�<�A�<ܒ�<�k���	���=�<������O�=Hl|<
��ذ<x=j� = =Fo�=��O�x	�< ha<*=��%=�{�<z>���<�Z^<F����<h��Xs�<�Ȼy�=f�9=���7*��o�<��o;��=�]<�{�?�=xH�<1BW=�د=`�N�"��=�y_<,F��՞<�w�<&�<���=��XÒ<\9�<��=J��?��=����5�<e�=�`ϻz�=4��<X?�<0D��䡼j'N=��=@�y�Ld���������;9��,���P� [�:�$D��|@�cB=&�M�@��h��=X�=芽�\�=�p�=)O=�K>��&:��C=z K���=��H� |�:Ey�=剄���6=�ٻ�H�<��=X~'���%��v�����:Z�v���r;�ռ�p����: M�;YL,=,+����D�@=��<t凼�@�;@��:X��<.km����=��=&���虼���<��qR�= �i�r�=���=���k��b;=�.=^{J=|�=�=��\<�ϼ�9����l����=ZJ=8�[��(��;�<H�=Hd<H��.�!��SM<���<Ȱ�< ֲ;�)��L 8�qC����0'�<�^=4X=)=�$�<0">��B=�}=��ټ���=�o�;\t�� ��;�#=�RB<�����d�h�ּ�cZ���>�>=J�;���<���<t��<a�=d�>�= ���0ʻX�Y=d��<	�u=���<� :�p�!�������=�;Լ� =�:0��  = p>�Hs_�n�f� ��;��<x茼�7=��<A�=�rt<�e�=P<Kx����#�Y�=HR�<���H��� �d;�"�0JA<��e=����Vkw=(�,<���3������c�dK=�а=V�>@I;���*����<Y3�^�<�-�6=h�x��-=�U= k�:4��<�+�=\N=����ȍ�<���<��;��a�ᆥ=�v=��ѼUH�E�M=`E�< �;��3�\��<R���O<�t���(=�b�<Xy����;��=(A=�A6=S��=Ҹ?�1Z<�;�<̤�<�W3=�M<=E�н��;��1=߼����<�s��m�< A)��u�=$C�<`j\���;<��<�/^���< �n<t���E�,=4u�<��P=��C=�x�;e��= ��<��z��=T�<��u=ԯ=(=�p�<=�L�P�r����= :�9��
=@+�=ZG#�E e=�P<I�;��ܻ����@.�x�M��j�:0���[`����h�����$%�(��ʁ=@S[<�ܼ �$���P�ֻ�H�<�#�<fI+��'=��=h;<��F=`~���-=��a���w�����B��1)=L�Լ ��Ȝ�<X�O��+i=�f�@�;�:L<���;n�����<������<��A; ;,;0�b��Fռ��n�z=�f�;��<���<��<<��<�
�P"?<j=�S�� \d���/���'����<��5<0=��<���6D�����m_<�ڭ<�Ԕ��Jw=h�a�����\<�?���r0<6�=Dݹ��-�`[�;Z�=��
������V� V<��z�Pp<�;ټx g��%;�cռF<�����f=P�<,��<��n;P��=x.�<���<`�);V�=@R[<P���������<0
�;,�Լh���@ ����,��L�=&=���<��;��<@W�;TO=�%8=�8Ǻ A���r_�p��<�=h�(= �ѻ��ɻ�e�<�R;�R�=xh���=�]��8��`��t�<���<�@p;��!=<���h�(=#�L=>=<��=ѝ=T��<�c��ܼ�l�<3E5= 6����V=r<����h�|�	��=f^=�?�=H�1���	=8j`� �g;�&�*?d=,FO=���=��m;(k��{���8�< � <�⑽��=���ib?=�hD=xU�<��v=m7�=�ғ<�Q�=`gA<�_�<�f�<��X�^f�=��<���� ��==�p@=�M�=��2��Ti�
y� �����y=��=8>g<6c�|�<��z=�A=Ԝ=�}�< �.��v�=��0=���;��PN��pڼ�&�1��=DL���ą=u��R�z=E/�����=�% ���P�L�g=���=��5;÷=Q�Y=$���p=_:=�)�=lX�<��.=k�=��<`EX<�P=���;z�
=�uf=x�<�
���<�<��ό���=(8���<D�i=(5��i	m=�Z�;_^=�v�=��L�G�������<��=|< ��94cＸm����ļ ��8�'�<ܑ
=���`�l;г=���<�;���;���U=�j���}�>Ҽ�;ּD)�<��=�
���@�d[�<��$��'=��q<�;$l9���;ԃ���ڻ9�>=�A���#�<���<��:?�@=8c}<z�=LD��X�u��` ��5<H�5���;H�<(b�<
�;p;�<@�=�*[=�>W�.��T���O� ���H�<׼�=���`w�����6�/� 8�;  Q�L��p��<*��ؙi<��U= �<��Ǽ���<}�9=�	@���� ��92� ���+< g.� Fd9<t��H�m<,}��a<h�p��hW�<��7�"= Ё:�O�;T��Pd=��< ������<�^1<UJ�=�8���*< _���.��=�<�~������̃�<,�=�c=�|V�<`K׻���#=`��<$`�����@��;��]��� H�E�+=~>^����=�3�<-�Y=T�:�`�U<d�P����k]�F��= �y< �;��{=@�:��b=֩�=8�^=®�=kMe=��(=�ڽ@�a���<���<��\Kj= �<���޼@6�2��=8cu=�j=p�<l�<0=���8=(
�<�`�=�>�<(�= ��<�	��5��4i=�|�<����(�=��;�0=���<O���4v="��=xg0<��=P_��\���Y9,=̛�<6?�=$��<����v�<���<p�=$@=bi ��<�_�,�<�η<�!~=�Ͱ��uۻxMv��y=vԛ=�+����<P9���Ԉ=b=���<}�8= �?����jG��.!=&�	��f�=�
����.=��{���$=<�����O�oc-=��f�a��=`��; 
*;d��< ��:��p=db�=�M�=��=d�<����*=��� G�;�܄< �Ĺr�=�D5�@�g��G���z=�FǼ���<>D
=p��,ΰ<�����,=�.b=�Rb;i���ڽ�0)�<&�=@b; 8��6��, ��ļ�m���<Y9=�1�;����C=YS<h�&�p��*�v����<�U �0�׻j�0�̼U�L�(=W|�=�ϓ���e��@h<L�=���#=0��;p�0<0v}� 8;8�G�����$==�2W�X�{<�#�<�`��W�_=��?<T�Z=΢'��e_� !%�   <�x���l�<0s<��<�gx<t��<����/=��ۻT�켰͕���g� <�=^�L=  �tN���;�*�V��.�;�@ü*^���<�������<�{#=`�{<�ţ�w�<�=�(���@��͊���㼰��;���� T��
��5J�
�O�<��X�h����bD<H�7����<��:@�û�w��@�<��<p�A�Sl= �<:[�m=��L��Jx<H�N��v����<l�� O8���<
d�=�.2�C�<8P,���ª<v�=�G�0r����A�E����8G������<K偽l�伫Pe=t��<��=���@%�: P���v<�`�����=j����R;��=P��ty�<���<+��=f�=0W=�H&=e3������ҋ=t�<TG?�`�%<�[+���n��߼ذ�=;�1=�R�<X�H<��<`�ڼ��k=�ق="�=��;�W�=8�z<������;�o=���T����=;�<���<0�L<vk��	=P��=P��;�$=��X�b���'=dn�<#c=�:�<n���%g�=l|<x=N�� �+9��3=��<|b�<zdR�_�=��8�H/h�.�½lJ�=��=j}���.==�����=�੻l�<P[�=p>�<GL��O񅽸��<~n}�� �<@���2�=�诼�C=a� =�d����x���쵼�(=$�缠�`;��#<�^W��x?=���=�I�=2��=h'�<�����< �n��Z��(0�<���;Jת=H���>=�ƽd�=RNE� 9`� 1z<0��;��ټ�:=��=�6�; ��:�蠽�nǽ ���Y`Q= I�;8/h<�{��s,;/ļ؎[����<,]�=h�0<�߭<,=N�= ���<��R�R��Nӻ��-=�Xм���$	��*~\=;
�=�������\��<�'/��U= �� it<P���˥;X�,��_J�o�=�2�� ��:��:=����k��=�1��4�A=�`?���3��������m�t� =@�F<��X<���<�d=�*��,q�<���:p�<к�$d��F<xH <r#=�����{��$�������F�:^�@�8>]���"=t������;i�<�x3=�xM��[=yR_=�|���8b��+�tp����=�������X���/�*��(�<&��q� �6��м�#y=��޻�JZ<��d� �a� �;t��j�<`�C;i�R=�<�&<�T/��}��b<hR�=#=:�<߸v=X���< H�;��ۼ���8�=Z#���E^�2M>�m���nS������\<.�-�d����lj=H�+<8`'=��� Y��@�;u�<���dt�={���{8<P]<8�<5�;h�<iڵ=d9=�ʌ=�=n�e��)=��Ђ=\�=�^<�������I�Rg���F�D)L=�j<���<��=�zλ��)��3=֓=(�<=`�<|��=X�w<*=���$<�l=0Y���^;��R=��U<pL�<�m<�����<B�>x�<���0���J���<ڄ�ܙ=b��=ƾ����y=���<�B�<¬_�@���^4=`D<�U;��|�Q$�=H������6���p=��=V	���8=� ��U%<������<S�=<2#=��ٽ���pCf<�I����S<8��d�<�!)���)=���=�H������O�å�h?�<�$��|s�����: �#�D�@=��=d�=�T�=[�<\ҝ�>y=0�<�%���	�=pQ<���=ɾx=+���K>��h��ݥ�<E=�_*;t��=`+l<|*.���������;� j���L��pX���Y;�� N�;0-Լhi���m�<&@=�Fݺ�s= H�;��<�Lƻ��<j_2��>��im= K�����<�!>�a	y=��=P���"�l�X�7<��;���< Ҁ���E<8�D��:=�0���7���h=0aû��˼/,=P9㼐�=Pyq��s�<@�N� :7��U< 1 �Pm��/�<�G@<�D;��$=谛<08&��< ���hʻ<(r<<��h��K�<�ۻ�Q�<��Ȼc������}�; �O<�,ۻ�͌�N�3=��(�8��8Xp<:�<@�����=<=���ܾ鼐�m������<Z�@�r�P�j0����;��ռ`�<��S�y�|ټ��n;0��=���:|�=�D�C<)=�s�<�O�� x;`�<H��<�_;��#<�K<����b���0l�H�=pIs���=��<@�P<�<�O�;`�!;��<�U���}��}8��QD�\�����b<,��<�,=� Mռ0��<dǳ��{�=Pd����q;`�T��<��h���=����� �<X�><@�; ��<PDW<`��=���<&b�=�^=�)%� 	��?�=uNP=b
��4Ƽ�f����X7�*L=�TG��T�<�w0=xP似�:��U�<�b�=�AN=�NX=ǿ�=x�n<���� /7:��U<X�[� �A��v=8=M��+1=�)�<��彘h�<#%>�*=����J`�P��H�<�Rڼ�l�=�%�=�ڌ��z=μ�=D�<�V�
��Q=`U�<��<Ĉ��=�ü��'���#���:=ʝ�=�Y�x�7=v�0������;��;��=�&T=����RM_�-�<������;����;�>'�f��=,چ=p�r���ZQ��R���<�����K�@+<�tT<��=;z�=a�=�f�= ��; �&:��6=��<�
=��=����)�(=�>�<���<v�]��+>��輐&r<�>�=@gJ���n<�U�<�^��J� ��}�� �й�m
�(������; u�:��!�� ����������>=@�=|������<\����7�<�=�w=�X� �p���]=��e�dښ=h����=�5]��%�;I����Y;�{[= ����E<4��{�j=V�L��bL=�ñ;"Q8�ݚW= �����$= �e�Â; hl��ǔ:�X�<L��<��<��]=�|M<�]�;�T=��,<D��<��<��^�@�;�J�<��(<y�=0���H��<乀<�|��x��$\�<�|
=���<�a;ո�=��,��@Ƽ B��ܟ<�<�=���;<�����ѻ �; �X���������Z��3�`.
<H�ڼ�  =�z��p�弘U���0<��= ��;��K=�I��Ґ�=��_=&�;PG�;|�[=�1?���`�HF!<��<@���t��@g`�a{%=��g�k�=�{"=0�D<� =a57= "9<��=�5�<�r�<.]�Ơ6���߻]�=H@<`�������0��;����N�=x�B���=/
�l�=�nU���X=b�W�hq	=�л<��;��m<@qJ<�M=H�c<���=���<�4�{���$p;��{=��r�<�������FB�0�;~v=P~ʼ0�=TS�<$�`�H��e;�z�<�=��Y=&e�=��J��}^������ٻK��¼} �=dK��l.P=�Q2=hq?��^�=z�>���<�^+�(T<��"=Hb�<�<���=�c�=���@덻��=s=���<j�N�\v�<<��� ;S<ȱ��~�=����NF�� +�V4p=��;=�P�|e=�M� �W9�zA��P¼��=��b=�n��`0ǼIΘ=���� ��@E]�l�<��/�:mx=�3*=@5+��Ӽdy��p��|c�<��1<<Q5���<��8���=��#=H:�<�> T��x�<A�J=Ob=R�=��y=�����2=�`�< 8�h�q�Bu">`�7;��<tr�=d��y�0= 4�;����D�5��������j
� �b;�1=$��<*���@�<�F��ګ=XG�=N �=@�ĺ �m;�<�ht=@R�:%�z= L��&�<��=��`����< �Q���=���<����&�����:8V�<]�=n
-���=@y��= ��@e��j�f= �4<@6���.�=�2�&�=���p���n�=��X�;D�
=��?=0'�;K:�=x�=\��<r�b=��3=l����<�kF��x=�4��P=5<H�<�<4�=r��|釽 �V<�r}<\��<(X<�ja���=� �P�6��z�;��=<ƼE�=�)=@�º��FK>�ʘ����<�ܽ`Z����ż��;��!��Dy=x|*����h�(��ۼ�5�=س���e=��O�=��O=��{<�
=���<��9�8I= �";�#<��|\=���vJ�=b����=��=���<dqI=�N|=@o�:ƉU=��0�X	��Y༕Ҏ�\���T=�c��%�ً<P��<��i����<p�9�Tz�=t�۽���,#����ּުL�`�,<��b=��<4츼��<l��<�
!=��L� ��9��� /����:g|�= ���^�<Z	�ci���eK=Ĥ�=8Y�x>���� ;P�<�n2=��f��Ŧ=�G�;Bl�=����'��[�@k�����o����<2P:�m=� U=�.*=�@t=:�!= /x:�U�=��=��=a/=�k�=�$�=�
<d��<l�^����=�V�=���=�\���m�;6�h��-*��}<85�<P���&� ��v��= �t:d��<WK=�_��=�o�3��� ��:О���C���>��Z���<k��)�=��～�Q=�Ζ��,��F�='D='�<:��=��K=�P��W,1=�:<��@�=8"���ND���=>o�=(�G���=��7=���<��=��[���<����0�T��<�1>�g�<�J���;=����k�=\���[����=j�������i�Hyr<�=@^z�����瘼����޼`�,;���<��x��c�����:_2= l�:ĵ�<����&1���P=�aּ\��<��4�X�0�H�\<.��=�u�L�����<����\��<`��<��R�`%�ܶ�<�$@�PO��0��;t>�<�<<0���(�<��<��><]�3=��;�̒��1��n�;�c�:���0����Y�:0� �~����I��=�0O�²e��B�@?T� m�;x^=s�z=�ͬ<h�<)���h���;��<����h�<���5�<,�=`���K�@��< ��<�a��%��� =@��G;����<�I�<P{���N�<�p󼸰(������;~�<�1W��#�<�k< ތ;�)��M=�
=�d�<0<��<A�c=�,�@�����; Yۼx�_�4vټ�ހ�X��<8f�=��P��v�<@�P��K�B�A=�c!<�X�<$A�<��_<G�� H˻�A"��.d=��$����p�<����1=�<�N�=X�^����P�>� ����o�P_�;�m�=N�=�/ ��Q�<,k=D Y=���X��<�����ʨ�d;�<��=�H���	=Jk�I總�*=ʏ=�E��d5�=X�g�  �8P�;Ɋ= �ɻ�&�=�v=N�:=�M)�Gw�d������<�H��b�|u�=�'2� �<w�"=H����L�<��=����2g�=y+Z=��U�ʆl=T��=��=�9�;TA<4�����=���=<��<L�;�"�<�U�������x<LF��8 �.ܵ�7,�=�� =؞��r%=�C��S�=L������x�=�%
��7���Ľ:��=����x{�<��߽�n�=�
���<�w�Z�|����;�6'� 'p�ː~=  :�|1�l=8��ݾ8= i*;�w=61>>�cF=�c��>�<��%=ܔټo� =\�C��j=x�o�<Ɠ�d�����=,��<j����=`���?M�=9����5����=�|A�B�_�bl��<{�<���=<G���\��6м�Q��:#�����۶<D���+�0��;~Q=�4�� n<:�a�U׽Jx�=����;9=��м��(�GD=�s�=p�y�26W�T{�<x�޼�Q�<Μ7= VH��jԼ�	=��<�ٜ���5<h�<�.�<�2���ć<(K�<�7�<���=���uN��S���)���>��żLC�w�����(;� n:s.�=�����򦟽B����g�<�@=���=fS&=2}<�&��8��(�g<8�<=F���
=�%���F=.=V�F����:t��<�:�<̡м� i��pX=T�������h&j<���;�}��@��<JN=��9ϼ0�{��T{��*�<0�����;���<��� jH���E=�Y= ��:��q<��<xH�=�;���t�;`KW<�&a��r�T�L���Ǽ!(=��> �{:�(R<�l3�T���-,g=@��<n�<h�'= ���f�0p̻��
�Nǵ=�T��e����<t����A=H���\�=�a�Pt}<d���lr�<�Ҳ�H�<,��<��'=�'b��V��l�=\��<,ʝ�w�#=� r�*6Խ�Jp=]}�=p?Ҽ 3��DŽ�6��e=\=~����=�M�t����.º���=��=�by=`�<H�7=���x�@���0q�<�N.�(H< e�=d�=�w1<�
=��2�l<���Ͷ=�<�.�=�A�<�;k���d=Ao!=|}�= cn;hw�x�M<GP�=Y0�=F�^�0v�<3S1=n�0��U:�;폽�31<�K}���4�~l�:��=$e�=
ӭ�%�,=�}�=��y�@z��>B�=���<��������=+Iؽ ���j½�[h=4G��@D&�@�<<>�a�W�����p��7� �r���*��4�6�<�8��H��<��
=�-1=`��=��N=v���b�< �I=s���@'=�j���=�����|{<������>/��=c�,(=X�!�XB==���<@*��<X=��g�v����� /G<�yy=�悔�;n�b��OW�$�"���{����<܃�8�p�hH�<��Q=��ռ(�4����K\�=���<��:=X���m��L=��=�⨼����-=ȳ��"�<ޢQ=�Y໨�(��=x�~��ʼ�/�<�3?<xh< 'y�X<�th=��+<��=�'��"��8�U�
T�~Ƽ�伌5�� �&�[�2���𐻧�=R�1�rR[�hl9�\ĝ�}$=��<ޢ�=�}=�qg��fν�����#�:h��X�U�sK$=��b�Z�=���<�Yl��˼��=��<����0����m=��E���J����<�:������k<'���2j��Խ�,�p�<�8�8Ү<���<,U�` ���(=��<Ȕ�������<�Ǟ=�x������П<ak���Nq�J��ܾ��7,4=��>T�<�/<Al�h�нp�=��<�|&<п�<:�.��i��X�X�����g��=
�?��m���J=[��D�]=�ȽU�[=�����H=8��d�V=�ӱ�,1�< �_��4=J�R�R�8�zp�=�$���<�d,=�֕�0n���7=�a�=�1� �8�p��)�ý 0=`��;FY@��{�<@������9;'�Q=a�&=�<=�ϻ5�K=T�������pр<HEa�S���9=�3u=J�=�UE<z�)='�?� ������=@�;�I�;����l��M�-=XH�<�[�=��=���0�<$0�=��=�;�� [���=�rۼ���ꤽd��<��@�������콍L�=�xr=@�ֽ ��<�5�@�z��&�Tㇼ^o�=�==���7潎+�=ʇ�8��r������<��@`��م=������ܽ�`��#�L3ͼn�1�ȣ�p�<H�|���r<�t=�_�<���=�y�;d1��)!=�N=�H����M=��<4Ϻ=�F�M&=�Ղ��0>��r����*<E=@�L�yF<d�<��漀�c<Po��p�������+~; . =E�入��=����0��P�����;����������<8p�<$)g�L(<!����������= 	��6��=��w<L��DA5=�������a��{<�R[<���l3x=l[���q�;T]_=@E�2�(��2�< f���E�� �;�/�;`�u<>��=�8�;(�%��֊���1�8wP�$����Lm��2��b]O��z�@́=m{�=�����ﺽ�E�)��l_9=`��;0`s=�E�=��$�o\������ v)�Ȇ=�F�<�K�< .���<��:<5�ڽ�K�< % <8���z�`�; ��=���;8j��t~�<�@�`�����<�{��>���7��������ġ���'Ļ\en=�[�ؑ�<��i=�H�<�,.�0�$�|U�<�&-=V�׽�2��=�W&�r%ý$;R���\�(�<g\>��*=�O���V�������=�<�ф=T[�=RTI�r�|���<H%H����=T5ļ�5��+�(=@���!�=fýO3D=�K ��mv=���&��=a׎�F�=��3��n=V�(��FH�?�4=�����<��,=@���G0�@�\�'��=��H��.0�~Ô����x��<�<�3���*=���;���M�<�u�<hv�<��S= ����{=R	��Wν���<� �0#��@H=^��=0Ղ;S*=?E@=����đ;t�=�ݦ<p�n�H�7<Ȗ��n�=̿3=�t�=��=��>�\�<��=��&=86Ѽb:��x�<t7��ܣ����~��>�<d<�d�ռ��νއ�=�k=>o��(�p<��s��K��������%�=�P9=X��������=0���h�4�i����<��������5{=�vɼ_nҽ��������l�
Z�8_5�P��;���S��tP�<��<F��=��{��j�%=5sa=��<�>2=�g\<���=@A�� =�<��E��'>>@;��|��o�=��|�$�<h�<�]�����<�!2���ܻ$�ƼhvF�8z<8ܵ���P�Fi���N��Y�Xp���+=>�w��w����<8+R<�)��C�<ؔ���b���p�=@b庒o=ܠ=|�ҼW�<=Z��0���2 i��]<D=�U���2=�y��X=;kc=����ϼ$\ �h�<tr߼�H�n��Ȍb<H��<䲀=(�"� 1���*0��_��X;85t<8(K��H��과n��d�(=��_=㡓��=�� �d��,r��@=䓅<�Y=�^�=H"3�"���1���<��n=��/<�r-=-�<�m�<��9�a���l=@׷��l���ʼ�"�;,e�= ��9�np��ϼ���;�:,���O<
J�֪��M(�𨹼Аü@H��)<8_=���<�<N<�=��>=��4<�殼�=X[<dQ���3�_=�_r�����|<��t��JY�C�=R=��C���Ec���=�
=��i=�X�=��$�"}���H<h2n<D�=���.�+���<���$�=�\��Q��=������=�_���f=f�W"=H3q<�;�<��J�YZ��}�<p:�`��<�`=�|ໞK�z{�4��=̟n��v�����q���a=�	=ZzV���=(�>��-���2=PAv���n��=�hȼ�>H=z�w������;�;4OC��i���;W�x=��rKN=�=�Y�B82=�L�=�p=(?��B9=�'G=�o0=�#�=$��=h��=�W;�pI����>�X�<�V=
�|��v;����e�ښ)�0���\�>��h6��\ҫ=�6=��G��\�<����`�e��������'Õ= ,�<�(��=��s�>6��$����΄�T=}�ݽ0�K<�<P	ʻVI��ב��j����;`(������p<<<Nn��@q��� � 
��i>��K�8�<�/*=Ǒ�=@N��giJ=p�[��=hNż�V@< �׻�kF>���<ơ�l�=@�����h=��[��ս/�m=~0��nhK���t̽���<8(��^��>'%��T���/<�q�<h�=h�/<�h|��J�;���<Pއ;�G�; ��f����=0.���^<���;"���$= �~�̏��z�n�P[�;fD=+�<��;`:h<��м��/=��\�p�4���q����<����~�<h�a�^�=���<�M7=�^R��C�;����k�<0��;��<���T���`dV�� `�p�ɼM�:=�ҍ����мh	;����<x��<�<=�=(�.���Q��<񼬚�<Y�8=�5ͼ��j=�hL<䬣<�8��P|H�@�<`,i;\<��������o<�\U�`�����K�@�|;ȱ��RW;&�L��s<�����W"�����hϼ �V<|= �;P̈́;��0=��Y=|P�< V{�P�E<�<<�=��(z6�,�<�C�������X�j<�����tB=n�=P��;���<`����ו<�0*=�]�;�< :ּ�5��`׻�"=���<�)_�@e�
�
siamese_3/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*&
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
 siamese_3/scala1/moments/SqueezeSqueezesiamese_3/scala1/moments/mean*
T0*
_output_shapes
:`*
squeeze_dims
 
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
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_3/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_3/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
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
Hsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_3/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
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
 siamese_3/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
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
Nsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( 
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
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_3/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
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
siamese_3/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_3/scala1/cond/pred_id*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`*
T0
�
siamese_3/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_3/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
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
siamese_3/scala2/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala2/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*
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
^
siamese_3/scala2/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_3/scala2/concatConcatV2siamese_3/scala2/Conv2Dsiamese_3/scala2/Conv2D_1siamese_3/scala2/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
�
siamese_3/scala2/AddAddsiamese_3/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese_3/scala2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese_3/scala2/moments/meanMeansiamese_3/scala2/Add/siamese_3/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
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
Nsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_3/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
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
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
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
Tsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Isiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_3/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
 siamese_3/scala2/batchnorm/mul_2Mulsiamese_3/scala2/cond/Mergesiamese_3/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
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
siamese_3/scala2/poll/MaxPoolMaxPoolsiamese_3/scala2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0
�
siamese_3/scala3/Conv2DConv2Dsiamese_3/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
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
Hsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
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
usiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
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
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
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
siamese_3/scala3/ReluRelu siamese_3/scala3/batchnorm/add_1*'
_output_shapes
:�*
T0
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
siamese_3/scala4/Conv2DConv2Dsiamese_3/scala4/splitsiamese_3/scala4/split_1*
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
!siamese_3/scala4/moments/varianceMean*siamese_3/scala4/moments/SquaredDifference3siamese_3/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_3/scala4/moments/SqueezeSqueezesiamese_3/scala4/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_3/scala4/moments/Squeeze_1Squeeze!siamese_3/scala4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_3/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_3/scala4/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
ksiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Csiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
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
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
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
(siamese_3/scala4/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0
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
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_3/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Nsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
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
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese_3/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
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
siamese_3/scala4/cond/Switch_2Switch"siamese_3/scala4/moments/Squeeze_1siamese_3/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_3/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
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
siamese_3/scala4/batchnorm/mulMul siamese_3/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_3/scala4/batchnorm/mul_1Mulsiamese_3/scala4/Addsiamese_3/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_3/scala4/batchnorm/mul_2Mulsiamese_3/scala4/cond/Mergesiamese_3/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
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
siamese_3/scala5/ConstConst*
dtype0*
_output_shapes
: *
value	B :
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
"siamese_3/scala5/split_1/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_3/scala5/Conv2D_1Conv2Dsiamese_3/scala5/split:1siamese_3/scala5/split_1:1*'
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
siamese_3/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/concatConcatV2siamese_3/scala5/Conv2Dsiamese_3/scala5/Conv2D_1siamese_3/scala5/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
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
score_1/Conv2D_1Conv2Dscore_1/split:1Const*
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
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*

Tidx0*
T0*
N*&
_output_shapes
:
�
adjust_1/Conv2DConv2Dscore_1/concatadjust/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:"�n-