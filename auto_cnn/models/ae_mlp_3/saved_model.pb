Δμ
Κ
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
<
Selu
features"T
activations"T"
Ttype:
2
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
Α
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Κτ
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
Δ*
shared_nameenc_2/kernel
o
 enc_2/kernel/Read/ReadVariableOpReadVariableOpenc_2/kernel* 
_output_shapes
:
Δ*
dtype0
m

enc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Δ*
shared_name
enc_2/bias
f
enc_2/bias/Read/ReadVariableOpReadVariableOp
enc_2/bias*
_output_shapes	
:Δ*
dtype0
u
enc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Δb*
shared_nameenc_3/kernel
n
 enc_3/kernel/Read/ReadVariableOpReadVariableOpenc_3/kernel*
_output_shapes
:	Δb*
dtype0
l

enc_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*
shared_name
enc_3/bias
e
enc_3/bias/Read/ReadVariableOpReadVariableOp
enc_3/bias*
_output_shapes
:b*
dtype0

hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:bS*&
shared_namehidden_layer_0/kernel

)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
_output_shapes

:bS*
dtype0
~
hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*$
shared_namehidden_layer_0/bias
w
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
_output_shapes
:S*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S
*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:S
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
Ί)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*υ(
valueλ(Bθ( Bα(

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Λ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Λ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
Λ

!kernel
"bias
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
¦

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
* 
J
0
1
2
3
!4
"5
*6
+7
28
39*
 
*0
+1
22
33*
* 
°
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
\V
VARIABLE_VALUEenc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

!0
"1*
* 
* 

Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
.
0
1
2
3
!4
"5*
'
0
1
2
3
4*

Y0
Z1*
* 
* 
* 

0
1*
* 
* 
* 
* 

0
1*
* 
* 
* 
* 

!0
"1*
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
	[total
	\count
]	variables
^	keras_api*
H
	_total
	`count
a
_fn_kwargs
b	variables
c	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

]	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

_0
`1*

b	variables*

serving_default_input_layerPlaceholder*(
_output_shapes
:?????????*
dtype0*
shape:?????????
ν
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biashidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_8818037
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp enc_3/kernel/Read/ReadVariableOpenc_3/bias/Read/ReadVariableOp)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
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
 __inference__traced_save_8818202
η
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biashidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/biastotalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_8818254£


τ
B__inference_enc_3_layer_call_and_return_conditional_losses_8818097

inputs1
matmul_readvariableop_resource:	Δb-
biasadd_readvariableop_resource:b
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Δb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????br
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????bP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????ba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????bw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Δ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Δ
 
_user_specified_nameinputs
Π

0__inference_hidden_layer_0_layer_call_fn_8818106

inputs
unknown:bS
	unknown_0:S
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????S*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????S`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????b: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????b
 
_user_specified_nameinputs
₯

φ
B__inference_enc_2_layer_call_and_return_conditional_losses_8818077

inputs2
matmul_readvariableop_resource:
Δ.
biasadd_readvariableop_resource:	Δ
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Δ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Δ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΔQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????Δb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????Δw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ά


/__inference_sequential_50_layer_call_fn_8817670
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:bS
	unknown_6:S
	unknown_7:S

	unknown_8:

identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
-

J__inference_sequential_50_layer_call_and_return_conditional_losses_8818010

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Δ4
%enc_2_biasadd_readvariableop_resource:	Δ7
$enc_3_matmul_readvariableop_resource:	Δb3
%enc_3_biasadd_readvariableop_resource:b?
-hidden_layer_0_matmul_readvariableop_resource:bS<
.hidden_layer_0_biasadd_readvariableop_resource:S=
+output_layer_matmul_readvariableop_resource:S
:
,output_layer_biasadd_readvariableop_resource:

identity’enc_1/BiasAdd/ReadVariableOp’enc_1/MatMul/ReadVariableOp’enc_2/BiasAdd/ReadVariableOp’enc_2/MatMul/ReadVariableOp’enc_3/BiasAdd/ReadVariableOp’enc_3/MatMul/ReadVariableOp’%hidden_layer_0/BiasAdd/ReadVariableOp’$hidden_layer_0/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Δ*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Δ*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Δ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Δb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:bS*
dtype0
hidden_layer_0/MatMulMatMulenc_3/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????S
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Sn
hidden_layer_0/SeluSeluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????S
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:S
*
dtype0
output_layer/MatMulMatMul!hidden_layer_0/Selu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????

NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
’

ό
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8818117

inputs0
matmul_readvariableop_resource:bS-
biasadd_readvariableop_resource:S
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:bS*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Sr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????SP
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Sa
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Sw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????b
 
_user_specified_nameinputs
Ά


/__inference_sequential_50_layer_call_fn_8817824
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:bS
	unknown_6:S
	unknown_7:S

	unknown_8:

identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
-

J__inference_sequential_50_layer_call_and_return_conditional_losses_8817971

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Δ4
%enc_2_biasadd_readvariableop_resource:	Δ7
$enc_3_matmul_readvariableop_resource:	Δb3
%enc_3_biasadd_readvariableop_resource:b?
-hidden_layer_0_matmul_readvariableop_resource:bS<
.hidden_layer_0_biasadd_readvariableop_resource:S=
+output_layer_matmul_readvariableop_resource:S
:
,output_layer_biasadd_readvariableop_resource:

identity’enc_1/BiasAdd/ReadVariableOp’enc_1/MatMul/ReadVariableOp’enc_2/BiasAdd/ReadVariableOp’enc_2/MatMul/ReadVariableOp’enc_3/BiasAdd/ReadVariableOp’enc_3/MatMul/ReadVariableOp’%hidden_layer_0/BiasAdd/ReadVariableOp’$hidden_layer_0/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Δ*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Δ*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Δ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Δb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:bS*
dtype0
hidden_layer_0/MatMulMatMulenc_3/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????S
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Sn
hidden_layer_0/SeluSeluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????S
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:S
*
dtype0
output_layer/MatMulMatMul!hidden_layer_0/Selu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????

NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
²%
α
 __inference__traced_save_8818202
file_prefix+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop+
'savev2_enc_3_kernel_read_readvariableop)
%savev2_enc_3_bias_read_readvariableop4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B€B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ϊ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop'savev2_enc_3_kernel_read_readvariableop%savev2_enc_3_bias_read_readvariableop0savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
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

identity_1Identity_1:output:0*v
_input_shapese
c: :
::
Δ:Δ:	Δb:b:bS:S:S
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
Δ:!

_output_shapes	
:Δ:%!

_output_shapes
:	Δb: 

_output_shapes
:b:$ 

_output_shapes

:bS: 

_output_shapes
:S:$	 

_output_shapes

:S
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ή9

#__inference__traced_restore_8818254
file_prefix1
assignvariableop_enc_1_kernel:
,
assignvariableop_1_enc_1_bias:	3
assignvariableop_2_enc_2_kernel:
Δ,
assignvariableop_3_enc_2_bias:	Δ2
assignvariableop_4_enc_3_kernel:	Δb+
assignvariableop_5_enc_3_bias:b:
(assignvariableop_6_hidden_layer_0_kernel:bS4
&assignvariableop_7_hidden_layer_0_bias:S8
&assignvariableop_8_output_layer_kernel:S
2
$assignvariableop_9_output_layer_bias:
#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B€B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ι
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
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
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_enc_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_enc_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp(assignvariableop_6_hidden_layer_0_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp&assignvariableop_7_hidden_layer_0_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: π
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
Μ

.__inference_output_layer_layer_call_fn_8818126

inputs
unknown:S

	unknown_0:

identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????S: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????S
 
_user_specified_nameinputs
₯

φ
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589

inputs2
matmul_readvariableop_resource:
Δ.
biasadd_readvariableop_resource:	Δ
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Δ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Δ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΔQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????Δb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????Δw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Α

'__inference_enc_3_layer_call_fn_8818086

inputs
unknown:	Δb
	unknown_0:b
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????b*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????b`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Δ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????Δ
 
_user_specified_nameinputs
9
χ	
"__inference__wrapped_model_8817554
input_layerF
2sequential_50_enc_1_matmul_readvariableop_resource:
B
3sequential_50_enc_1_biasadd_readvariableop_resource:	F
2sequential_50_enc_2_matmul_readvariableop_resource:
ΔB
3sequential_50_enc_2_biasadd_readvariableop_resource:	ΔE
2sequential_50_enc_3_matmul_readvariableop_resource:	ΔbA
3sequential_50_enc_3_biasadd_readvariableop_resource:bM
;sequential_50_hidden_layer_0_matmul_readvariableop_resource:bSJ
<sequential_50_hidden_layer_0_biasadd_readvariableop_resource:SK
9sequential_50_output_layer_matmul_readvariableop_resource:S
H
:sequential_50_output_layer_biasadd_readvariableop_resource:

identity’*sequential_50/enc_1/BiasAdd/ReadVariableOp’)sequential_50/enc_1/MatMul/ReadVariableOp’*sequential_50/enc_2/BiasAdd/ReadVariableOp’)sequential_50/enc_2/MatMul/ReadVariableOp’*sequential_50/enc_3/BiasAdd/ReadVariableOp’)sequential_50/enc_3/MatMul/ReadVariableOp’3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp’2sequential_50/hidden_layer_0/MatMul/ReadVariableOp’1sequential_50/output_layer/BiasAdd/ReadVariableOp’0sequential_50/output_layer/MatMul/ReadVariableOp
)sequential_50/enc_1/MatMul/ReadVariableOpReadVariableOp2sequential_50_enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sequential_50/enc_1/MatMulMatMulinput_layer1sequential_50/enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
*sequential_50/enc_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_50_enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
sequential_50/enc_1/BiasAddBiasAdd$sequential_50/enc_1/MatMul:product:02sequential_50/enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????y
sequential_50/enc_1/ReluRelu$sequential_50/enc_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
)sequential_50/enc_2/MatMul/ReadVariableOpReadVariableOp2sequential_50_enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Δ*
dtype0²
sequential_50/enc_2/MatMulMatMul&sequential_50/enc_1/Relu:activations:01sequential_50/enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δ
*sequential_50/enc_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_50_enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Δ*
dtype0³
sequential_50/enc_2/BiasAddBiasAdd$sequential_50/enc_2/MatMul:product:02sequential_50/enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Δy
sequential_50/enc_2/ReluRelu$sequential_50/enc_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Δ
)sequential_50/enc_3/MatMul/ReadVariableOpReadVariableOp2sequential_50_enc_3_matmul_readvariableop_resource*
_output_shapes
:	Δb*
dtype0±
sequential_50/enc_3/MatMulMatMul&sequential_50/enc_2/Relu:activations:01sequential_50/enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
*sequential_50/enc_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_50_enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0²
sequential_50/enc_3/BiasAddBiasAdd$sequential_50/enc_3/MatMul:product:02sequential_50/enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????bx
sequential_50/enc_3/ReluRelu$sequential_50/enc_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b?
2sequential_50/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp;sequential_50_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:bS*
dtype0Γ
#sequential_50/hidden_layer_0/MatMulMatMul&sequential_50/enc_3/Relu:activations:0:sequential_50/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????S¬
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp<sequential_50_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0Ν
$sequential_50/hidden_layer_0/BiasAddBiasAdd-sequential_50/hidden_layer_0/MatMul:product:0;sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????S
!sequential_50/hidden_layer_0/SeluSelu-sequential_50/hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Sͺ
0sequential_50/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_50_output_layer_matmul_readvariableop_resource*
_output_shapes

:S
*
dtype0Θ
!sequential_50/output_layer/MatMulMatMul/sequential_50/hidden_layer_0/Selu:activations:08sequential_50/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
¨
1sequential_50/output_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_50_output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Η
"sequential_50/output_layer/BiasAddBiasAdd+sequential_50/output_layer/MatMul:product:09sequential_50/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

"sequential_50/output_layer/SoftmaxSoftmax+sequential_50/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
{
IdentityIdentity,sequential_50/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
£
NoOpNoOp+^sequential_50/enc_1/BiasAdd/ReadVariableOp*^sequential_50/enc_1/MatMul/ReadVariableOp+^sequential_50/enc_2/BiasAdd/ReadVariableOp*^sequential_50/enc_2/MatMul/ReadVariableOp+^sequential_50/enc_3/BiasAdd/ReadVariableOp*^sequential_50/enc_3/MatMul/ReadVariableOp4^sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp3^sequential_50/hidden_layer_0/MatMul/ReadVariableOp2^sequential_50/output_layer/BiasAdd/ReadVariableOp1^sequential_50/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2X
*sequential_50/enc_1/BiasAdd/ReadVariableOp*sequential_50/enc_1/BiasAdd/ReadVariableOp2V
)sequential_50/enc_1/MatMul/ReadVariableOp)sequential_50/enc_1/MatMul/ReadVariableOp2X
*sequential_50/enc_2/BiasAdd/ReadVariableOp*sequential_50/enc_2/BiasAdd/ReadVariableOp2V
)sequential_50/enc_2/MatMul/ReadVariableOp)sequential_50/enc_2/MatMul/ReadVariableOp2X
*sequential_50/enc_3/BiasAdd/ReadVariableOp*sequential_50/enc_3/BiasAdd/ReadVariableOp2V
)sequential_50/enc_3/MatMul/ReadVariableOp)sequential_50/enc_3/MatMul/ReadVariableOp2j
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp2h
2sequential_50/hidden_layer_0/MatMul/ReadVariableOp2sequential_50/hidden_layer_0/MatMul/ReadVariableOp2f
1sequential_50/output_layer/BiasAdd/ReadVariableOp1sequential_50/output_layer/BiasAdd/ReadVariableOp2d
0sequential_50/output_layer/MatMul/ReadVariableOp0sequential_50/output_layer/MatMul/ReadVariableOp:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer

ϋ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817882
input_layer!
enc_1_8817856:

enc_1_8817858:	!
enc_2_8817861:
Δ
enc_2_8817863:	Δ 
enc_3_8817866:	Δb
enc_3_8817868:b(
hidden_layer_0_8817871:bS$
hidden_layer_0_8817873:S&
output_layer_8817876:S
"
output_layer_8817878:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallν
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_8817856enc_1_8817858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8817861enc_2_8817863*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Δ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8817866enc_3_8817868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????b*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0hidden_layer_0_8817871hidden_layer_0_8817873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????S*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_8817876output_layer_8817878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
φ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer

φ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817776

inputs!
enc_1_8817750:

enc_1_8817752:	!
enc_2_8817755:
Δ
enc_2_8817757:	Δ 
enc_3_8817760:	Δb
enc_3_8817762:b(
hidden_layer_0_8817765:bS$
hidden_layer_0_8817767:S&
output_layer_8817770:S
"
output_layer_8817772:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallθ
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_8817750enc_1_8817752*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8817755enc_2_8817757*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Δ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8817760enc_3_8817762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????b*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0hidden_layer_0_8817765hidden_layer_0_8817767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????S*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_8817770output_layer_8817772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
φ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

φ
B__inference_enc_1_layer_call_and_return_conditional_losses_8818057

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


τ
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606

inputs1
matmul_readvariableop_resource:	Δb-
biasadd_readvariableop_resource:b
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Δb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????br
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????bP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????ba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????bw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Δ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Δ
 
_user_specified_nameinputs

φ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817647

inputs!
enc_1_8817573:

enc_1_8817575:	!
enc_2_8817590:
Δ
enc_2_8817592:	Δ 
enc_3_8817607:	Δb
enc_3_8817609:b(
hidden_layer_0_8817624:bS$
hidden_layer_0_8817626:S&
output_layer_8817641:S
"
output_layer_8817643:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallθ
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_8817573enc_1_8817575*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8817590enc_2_8817592*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Δ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8817607enc_3_8817609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????b*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0hidden_layer_0_8817624hidden_layer_0_8817626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????S*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_8817641output_layer_8817643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
φ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

φ
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

ϊ
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640

inputs0
matmul_readvariableop_resource:S
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????S: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????S
 
_user_specified_nameinputs
§

ϋ
/__inference_sequential_50_layer_call_fn_8817932

inputs
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:bS
	unknown_6:S
	unknown_7:S

	unknown_8:

identity’StatefulPartitionedCallΗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
’

ό
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623

inputs0
matmul_readvariableop_resource:bS-
biasadd_readvariableop_resource:S
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:bS*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Sr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????SP
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Sa
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Sw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????b
 
_user_specified_nameinputs
₯

ϊ
I__inference_output_layer_layer_call_and_return_conditional_losses_8818137

inputs0
matmul_readvariableop_resource:S
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????S: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????S
 
_user_specified_nameinputs
Ε

'__inference_enc_2_layer_call_fn_8818066

inputs
unknown:
Δ
	unknown_0:	Δ
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Δ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????Δ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


φ
%__inference_signature_wrapper_8818037
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:bS
	unknown_6:S
	unknown_7:S

	unknown_8:

identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_8817554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer

ϋ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817853
input_layer!
enc_1_8817827:

enc_1_8817829:	!
enc_2_8817832:
Δ
enc_2_8817834:	Δ 
enc_3_8817837:	Δb
enc_3_8817839:b(
hidden_layer_0_8817842:bS$
hidden_layer_0_8817844:S&
output_layer_8817847:S
"
output_layer_8817849:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallν
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_8817827enc_1_8817829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8817832enc_2_8817834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Δ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_2_layer_call_and_return_conditional_losses_8817589
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8817837enc_3_8817839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????b*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_3_layer_call_and_return_conditional_losses_8817606«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0hidden_layer_0_8817842hidden_layer_0_8817844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????S*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8817623¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0output_layer_8817847output_layer_8817849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8817640|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
φ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
Ε

'__inference_enc_1_layer_call_fn_8818046

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_1_layer_call_and_return_conditional_losses_8817572p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
§

ϋ
/__inference_sequential_50_layer_call_fn_8817907

inputs
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:bS
	unknown_6:S
	unknown_7:S

	unknown_8:

identity’StatefulPartitionedCallΗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Έ
serving_default€
D
input_layer5
serving_default_input_layer:0?????????@
output_layer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:Όf
©
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
ΰ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ΰ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
ΰ

!kernel
"bias
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
»

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
f
0
1
2
3
!4
"5
*6
+7
28
39"
trackable_list_wrapper
<
*0
+1
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_50_layer_call_fn_8817670
/__inference_sequential_50_layer_call_fn_8817907
/__inference_sequential_50_layer_call_fn_8817932
/__inference_sequential_50_layer_call_fn_8817824ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
φ2σ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817971
J__inference_sequential_50_layer_call_and_return_conditional_losses_8818010
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817853
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817882ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΡBΞ
"__inference__wrapped_model_8817554input_layer"
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
annotationsͺ *
 
,
?serving_default"
signature_map
 :
2enc_1/kernel
:2
enc_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_1_layer_call_fn_8818046’
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
annotationsͺ *
 
μ2ι
B__inference_enc_1_layer_call_and_return_conditional_losses_8818057’
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
annotationsͺ *
 
 :
Δ2enc_2/kernel
:Δ2
enc_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_2_layer_call_fn_8818066’
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
annotationsͺ *
 
μ2ι
B__inference_enc_2_layer_call_and_return_conditional_losses_8818077’
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
annotationsͺ *
 
:	Δb2enc_3/kernel
:b2
enc_3/bias
 "
trackable_dict_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_3_layer_call_fn_8818086’
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
annotationsͺ *
 
μ2ι
B__inference_enc_3_layer_call_and_return_conditional_losses_8818097’
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
annotationsͺ *
 
':%bS2hidden_layer_0/kernel
!:S2hidden_layer_0/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_hidden_layer_0_layer_call_fn_8818106’
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
annotationsͺ *
 
υ2ς
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8818117’
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
annotationsͺ *
 
%:#S
2output_layer/kernel
:
2output_layer/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_output_layer_layer_call_fn_8818126’
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
annotationsͺ *
 
σ2π
I__inference_output_layer_layer_call_and_return_conditional_losses_8818137’
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
annotationsͺ *
 
J
0
1
2
3
!4
"5"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΠBΝ
%__inference_signature_wrapper_8818037input_layer"
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
annotationsͺ *
 
.
0
1"
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
0
1"
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
!0
"1"
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
	[total
	\count
]	variables
^	keras_api"
_tf_keras_metric
^
	_total
	`count
a
_fn_kwargs
b	variables
c	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
[0
\1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
-
b	variables"
_generic_user_object§
"__inference__wrapped_model_8817554
!"*+235’2
+’(
&#
input_layer?????????
ͺ ";ͺ8
6
output_layer&#
output_layer?????????
€
B__inference_enc_1_layer_call_and_return_conditional_losses_8818057^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 |
'__inference_enc_1_layer_call_fn_8818046Q0’-
&’#
!
inputs?????????
ͺ "?????????€
B__inference_enc_2_layer_call_and_return_conditional_losses_8818077^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????Δ
 |
'__inference_enc_2_layer_call_fn_8818066Q0’-
&’#
!
inputs?????????
ͺ "?????????Δ£
B__inference_enc_3_layer_call_and_return_conditional_losses_8818097]!"0’-
&’#
!
inputs?????????Δ
ͺ "%’"

0?????????b
 {
'__inference_enc_3_layer_call_fn_8818086P!"0’-
&’#
!
inputs?????????Δ
ͺ "?????????b«
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8818117\*+/’,
%’"
 
inputs?????????b
ͺ "%’"

0?????????S
 
0__inference_hidden_layer_0_layer_call_fn_8818106O*+/’,
%’"
 
inputs?????????b
ͺ "?????????S©
I__inference_output_layer_layer_call_and_return_conditional_losses_8818137\23/’,
%’"
 
inputs?????????S
ͺ "%’"

0?????????

 
.__inference_output_layer_layer_call_fn_8818126O23/’,
%’"
 
inputs?????????S
ͺ "?????????
ΐ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817853r
!"*+23=’:
3’0
&#
input_layer?????????
p 

 
ͺ "%’"

0?????????

 ΐ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817882r
!"*+23=’:
3’0
&#
input_layer?????????
p

 
ͺ "%’"

0?????????

 »
J__inference_sequential_50_layer_call_and_return_conditional_losses_8817971m
!"*+238’5
.’+
!
inputs?????????
p 

 
ͺ "%’"

0?????????

 »
J__inference_sequential_50_layer_call_and_return_conditional_losses_8818010m
!"*+238’5
.’+
!
inputs?????????
p

 
ͺ "%’"

0?????????

 
/__inference_sequential_50_layer_call_fn_8817670e
!"*+23=’:
3’0
&#
input_layer?????????
p 

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8817824e
!"*+23=’:
3’0
&#
input_layer?????????
p

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8817907`
!"*+238’5
.’+
!
inputs?????????
p 

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8817932`
!"*+238’5
.’+
!
inputs?????????
p

 
ͺ "?????????
Ή
%__inference_signature_wrapper_8818037
!"*+23D’A
’ 
:ͺ7
5
input_layer&#
input_layer?????????";ͺ8
6
output_layer&#
output_layer?????????
