Íø
»
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ö
v
enc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameenc_1/kernel
o
 enc_1/kernel/Read/ReadVariableOpReadVariableOpenc_1/kernel* 
_output_shapes
:
*
dtype0
m

enc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
enc_1/bias
f
enc_1/bias/Read/ReadVariableOpReadVariableOp
enc_1/bias*
_output_shapes	
:*
dtype0
v
enc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*
shared_nameenc_2/kernel
o
 enc_2/kernel/Read/ReadVariableOpReadVariableOpenc_2/kernel* 
_output_shapes
:
Ä*
dtype0
m

enc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*
shared_name
enc_2/bias
f
enc_2/bias/Read/ReadVariableOpReadVariableOp
enc_2/bias*
_output_shapes	
:Ä*
dtype0

hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä±*&
shared_namehidden_layer_0/kernel

)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel* 
_output_shapes
:
Ä±*
dtype0

hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:±*$
shared_namehidden_layer_0/bias
x
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
_output_shapes	
:±*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	±
*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	±
*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ã"
value¹"B¶" B¯"
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Ë

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Ë

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
¦

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
* 
<
0
1
2
3
 4
!5
(6
)7*
 
 0
!1
(2
)3*
* 
°
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

5serving_default* 
\V
VARIABLE_VALUEenc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 

@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
 
0
1
2
3*
 
0
1
2
3*

J0
K1*
* 
* 
* 

0
1*
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ltotal
	Mcount
N	variables
O	keras_api*
H
	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

N	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

P0
Q1*

S	variables*

serving_default_input_layerPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ì
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biashidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1415
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
½
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_1554
È
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biashidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_1600É
¢

ó
?__inference_enc_2_layer_call_and_return_conditional_losses_1051

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
Â
)__inference_sequential_layer_call_fn_1307

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:
Ä±
	unknown_4:	±
	unknown_5:	±

	unknown_6:

identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

ó
?__inference_enc_1_layer_call_and_return_conditional_losses_1435

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

-__inference_hidden_layer_0_layer_call_fn_1464

inputs
unknown:
Ä±
	unknown_0:	±
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¯%
Õ
D__inference_sequential_layer_call_and_return_conditional_losses_1360

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	ÄA
-hidden_layer_0_matmul_readvariableop_resource:
Ä±=
.hidden_layer_0_biasadd_readvariableop_resource:	±>
+output_layer_matmul_readvariableop_resource:	±
:
,output_layer_biasadd_readvariableop_resource:

identity¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢%hidden_layer_0/BiasAdd/ReadVariableOp¢$hidden_layer_0/MatMul/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
Ä±*
dtype0
hidden_layer_0/MatMulMatMulenc_2/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:±*
dtype0¤
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±o
hidden_layer_0/TanhTanhhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	±
*
dtype0
output_layer/MatMulMatMulhidden_layer_0/Tanh:y:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ü
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1475

inputs2
matmul_readvariableop_resource:
Ä±.
biasadd_readvariableop_resource:	±
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä±*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:±*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
Á"

__inference__traced_save_1554
file_prefix+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ã
value¹B¶B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ¨
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop0savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*i
_input_shapesX
V: :
::
Ä:Ä:
Ä±:±:	±
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
Ä:!

_output_shapes	
:Ä:&"
 
_output_shapes
:
Ä±:!

_output_shapes	
:±:%!

_output_shapes
:	±
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¿

$__inference_enc_1_layer_call_fn_1424

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_1_layer_call_and_return_conditional_losses_1034p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ø
F__inference_output_layer_layer_call_and_return_conditional_losses_1085

inputs1
matmul_readvariableop_resource:	±
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	±
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ±: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 
_user_specified_nameinputs
¢

ó
?__inference_enc_1_layer_call_and_return_conditional_losses_1034

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_1262
input_layer

enc_1_1241:


enc_1_1243:	

enc_2_1246:
Ä

enc_2_1248:	Ä'
hidden_layer_0_1251:
Ä±"
hidden_layer_0_1253:	±$
output_layer_1256:	±

output_layer_1258:

identity¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallä
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layer
enc_1_1241
enc_1_1243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_1_layer_call_and_return_conditional_losses_1034ÿ
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0
enc_2_1246
enc_2_1248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_2_layer_call_and_return_conditional_losses_1051£
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0hidden_layer_0_1251hidden_layer_0_1253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_1256output_layer_1258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_1085|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¢

ó
?__inference_enc_2_layer_call_and_return_conditional_losses_1455

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ü
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068

inputs2
matmul_readvariableop_resource:
Ä±.
biasadd_readvariableop_resource:	±
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä±*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:±*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
­	
À
"__inference_signature_wrapper_1415
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:
Ä±
	unknown_4:	±
	unknown_5:	±

	unknown_6:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_1016o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
û
ý
D__inference_sequential_layer_call_and_return_conditional_losses_1198

inputs

enc_1_1177:


enc_1_1179:	

enc_2_1182:
Ä

enc_2_1184:	Ä'
hidden_layer_0_1187:
Ä±"
hidden_layer_0_1189:	±$
output_layer_1192:	±

output_layer_1194:

identity¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallß
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputs
enc_1_1177
enc_1_1179*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_1_layer_call_and_return_conditional_losses_1034ÿ
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0
enc_2_1182
enc_2_1184*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_2_layer_call_and_return_conditional_losses_1051£
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0hidden_layer_0_1187hidden_layer_0_1189*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_1192output_layer_1194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_1085|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_1286
input_layer

enc_1_1265:


enc_1_1267:	

enc_2_1270:
Ä

enc_2_1272:	Ä'
hidden_layer_0_1275:
Ä±"
hidden_layer_0_1277:	±$
output_layer_1280:	±

output_layer_1282:

identity¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallä
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layer
enc_1_1265
enc_1_1267*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_1_layer_call_and_return_conditional_losses_1034ÿ
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0
enc_2_1270
enc_2_1272*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_2_layer_call_and_return_conditional_losses_1051£
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0hidden_layer_0_1275hidden_layer_0_1277*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_1280output_layer_1282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_1085|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
æ,
å
__inference__wrapped_model_1016
input_layerC
/sequential_enc_1_matmul_readvariableop_resource:
?
0sequential_enc_1_biasadd_readvariableop_resource:	C
/sequential_enc_2_matmul_readvariableop_resource:
Ä?
0sequential_enc_2_biasadd_readvariableop_resource:	ÄL
8sequential_hidden_layer_0_matmul_readvariableop_resource:
Ä±H
9sequential_hidden_layer_0_biasadd_readvariableop_resource:	±I
6sequential_output_layer_matmul_readvariableop_resource:	±
E
7sequential_output_layer_biasadd_readvariableop_resource:

identity¢'sequential/enc_1/BiasAdd/ReadVariableOp¢&sequential/enc_1/MatMul/ReadVariableOp¢'sequential/enc_2/BiasAdd/ReadVariableOp¢&sequential/enc_2/MatMul/ReadVariableOp¢0sequential/hidden_layer_0/BiasAdd/ReadVariableOp¢/sequential/hidden_layer_0/MatMul/ReadVariableOp¢.sequential/output_layer/BiasAdd/ReadVariableOp¢-sequential/output_layer/MatMul/ReadVariableOp
&sequential/enc_1/MatMul/ReadVariableOpReadVariableOp/sequential_enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sequential/enc_1/MatMulMatMulinput_layer.sequential/enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/enc_1/BiasAdd/ReadVariableOpReadVariableOp0sequential_enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
sequential/enc_1/BiasAddBiasAdd!sequential/enc_1/MatMul:product:0/sequential/enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
sequential/enc_1/ReluRelu!sequential/enc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/enc_2/MatMul/ReadVariableOpReadVariableOp/sequential_enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0©
sequential/enc_2/MatMulMatMul#sequential/enc_1/Relu:activations:0.sequential/enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
'sequential/enc_2/BiasAdd/ReadVariableOpReadVariableOp0sequential_enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0ª
sequential/enc_2/BiasAddBiasAdd!sequential/enc_2/MatMul:product:0/sequential/enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
sequential/enc_2/ReluRelu!sequential/enc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄª
/sequential/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
Ä±*
dtype0»
 sequential/hidden_layer_0/MatMulMatMul#sequential/enc_2/Relu:activations:07sequential/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±§
0sequential/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:±*
dtype0Å
!sequential/hidden_layer_0/BiasAddBiasAdd*sequential/hidden_layer_0/MatMul:product:08sequential/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
sequential/hidden_layer_0/TanhTanh*sequential/hidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±¥
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes
:	±
*
dtype0µ
sequential/output_layer/MatMulMatMul"sequential/hidden_layer_0/Tanh:y:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¾
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential/output_layer/SoftmaxSoftmax(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
IdentityIdentity)sequential/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
NoOpNoOp(^sequential/enc_1/BiasAdd/ReadVariableOp'^sequential/enc_1/MatMul/ReadVariableOp(^sequential/enc_2/BiasAdd/ReadVariableOp'^sequential/enc_2/MatMul/ReadVariableOp1^sequential/hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/hidden_layer_0/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2R
'sequential/enc_1/BiasAdd/ReadVariableOp'sequential/enc_1/BiasAdd/ReadVariableOp2P
&sequential/enc_1/MatMul/ReadVariableOp&sequential/enc_1/MatMul/ReadVariableOp2R
'sequential/enc_2/BiasAdd/ReadVariableOp'sequential/enc_2/BiasAdd/ReadVariableOp2P
&sequential/enc_2/MatMul/ReadVariableOp&sequential/enc_2/MatMul/ReadVariableOp2d
0sequential/hidden_layer_0/BiasAdd/ReadVariableOp0sequential/hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_0/MatMul/ReadVariableOp/sequential/hidden_layer_0/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
Ê	
Â
)__inference_sequential_layer_call_fn_1328

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:
Ä±
	unknown_4:	±
	unknown_5:	±

	unknown_6:

identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	
Ç
)__inference_sequential_layer_call_fn_1111
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:
Ä±
	unknown_4:	±
	unknown_5:	±

	unknown_6:

identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¿

$__inference_enc_2_layer_call_fn_1444

inputs
unknown:
Ä
	unknown_0:	Ä
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_2_layer_call_and_return_conditional_losses_1051p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
2

 __inference__traced_restore_1600
file_prefix1
assignvariableop_enc_1_kernel:
,
assignvariableop_1_enc_1_bias:	3
assignvariableop_2_enc_2_kernel:
Ä,
assignvariableop_3_enc_2_bias:	Ä<
(assignvariableop_4_hidden_layer_0_kernel:
Ä±5
&assignvariableop_5_hidden_layer_0_bias:	±9
&assignvariableop_6_output_layer_kernel:	±
2
$assignvariableop_7_output_layer_bias:
"
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ã
value¹B¶B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_enc_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_enc_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_enc_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_enc_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_0_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¯%
Õ
D__inference_sequential_layer_call_and_return_conditional_losses_1392

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	ÄA
-hidden_layer_0_matmul_readvariableop_resource:
Ä±=
.hidden_layer_0_biasadd_readvariableop_resource:	±>
+output_layer_matmul_readvariableop_resource:	±
:
,output_layer_biasadd_readvariableop_resource:

identity¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢%hidden_layer_0/BiasAdd/ReadVariableOp¢$hidden_layer_0/MatMul/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
Ä±*
dtype0
hidden_layer_0/MatMulMatMulenc_2/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:±*
dtype0¤
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±o
hidden_layer_0/TanhTanhhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	±
*
dtype0
output_layer/MatMulMatMulhidden_layer_0/Tanh:y:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

+__inference_output_layer_layer_call_fn_1484

inputs
unknown:	±

	unknown_0:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_1085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ±: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 
_user_specified_nameinputs
¦

ø
F__inference_output_layer_layer_call_and_return_conditional_losses_1495

inputs1
matmul_readvariableop_resource:	±
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	±
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ±: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 
_user_specified_nameinputs
û
ý
D__inference_sequential_layer_call_and_return_conditional_losses_1092

inputs

enc_1_1035:


enc_1_1037:	

enc_2_1052:
Ä

enc_2_1054:	Ä'
hidden_layer_0_1069:
Ä±"
hidden_layer_0_1071:	±$
output_layer_1086:	±

output_layer_1088:

identity¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallß
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputs
enc_1_1035
enc_1_1037*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_1_layer_call_and_return_conditional_losses_1034ÿ
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0
enc_2_1052
enc_2_1054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_enc_2_layer_call_and_return_conditional_losses_1051£
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0hidden_layer_0_1069hidden_layer_0_1071*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1068£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_1086output_layer_1088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_1085|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	
Ç
)__inference_sequential_layer_call_fn_1238
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:
Ä±
	unknown_4:	±
	unknown_5:	±

	unknown_6:

identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
D
input_layer5
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿ@
output_layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ÈW

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
à

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
à

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
»

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
X
0
1
2
3
 4
!5
(6
)7"
trackable_list_wrapper
<
 0
!1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_sequential_layer_call_fn_1111
)__inference_sequential_layer_call_fn_1307
)__inference_sequential_layer_call_fn_1328
)__inference_sequential_layer_call_fn_1238À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_1360
D__inference_sequential_layer_call_and_return_conditional_losses_1392
D__inference_sequential_layer_call_and_return_conditional_losses_1262
D__inference_sequential_layer_call_and_return_conditional_losses_1286À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÎBË
__inference__wrapped_model_1016input_layer"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
5serving_default"
signature_map
 :
2enc_1/kernel
:2
enc_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_enc_1_layer_call_fn_1424¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_enc_1_layer_call_and_return_conditional_losses_1435¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :
Ä2enc_2/kernel
:Ä2
enc_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_enc_2_layer_call_fn_1444¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_enc_2_layer_call_and_return_conditional_losses_1455¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'
Ä±2hidden_layer_0/kernel
": ±2hidden_layer_0/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_hidden_layer_0_layer_call_fn_1464¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1475¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$	±
2output_layer/kernel
:
2output_layer/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_output_layer_layer_call_fn_1484¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_output_layer_layer_call_and_return_conditional_losses_1495¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÍBÊ
"__inference_signature_wrapper_1415input_layer"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
^
	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
-
S	variables"
_generic_user_object¡
__inference__wrapped_model_1016~ !()5¢2
+¢(
&#
input_layerÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ
¡
?__inference_enc_1_layer_call_and_return_conditional_losses_1435^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
$__inference_enc_1_layer_call_fn_1424Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
?__inference_enc_2_layer_call_and_return_conditional_losses_1455^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 y
$__inference_enc_2_layer_call_fn_1444Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄª
H__inference_hidden_layer_0_layer_call_and_return_conditional_losses_1475^ !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ±
 
-__inference_hidden_layer_0_layer_call_fn_1464Q !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "ÿÿÿÿÿÿÿÿÿ±§
F__inference_output_layer_layer_call_and_return_conditional_losses_1495]()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ±
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_output_layer_layer_call_fn_1484P()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ±
ª "ÿÿÿÿÿÿÿÿÿ
¸
D__inference_sequential_layer_call_and_return_conditional_losses_1262p !()=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
D__inference_sequential_layer_call_and_return_conditional_losses_1286p !()=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ³
D__inference_sequential_layer_call_and_return_conditional_losses_1360k !()8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ³
D__inference_sequential_layer_call_and_return_conditional_losses_1392k !()8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
)__inference_sequential_layer_call_fn_1111c !()=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_sequential_layer_call_fn_1238c !()=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_sequential_layer_call_fn_1307^ !()8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_sequential_layer_call_fn_1328^ !()8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
´
"__inference_signature_wrapper_1415 !()D¢A
¢ 
:ª7
5
input_layer&#
input_layerÿÿÿÿÿÿÿÿÿ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ
