ώΡ
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ΣΎ
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
t
enc_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:b1*
shared_nameenc_4/kernel
m
 enc_4/kernel/Read/ReadVariableOpReadVariableOpenc_4/kernel*
_output_shapes

:b1*
dtype0
l

enc_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*
shared_name
enc_4/bias
e
enc_4/bias/Read/ReadVariableOpReadVariableOp
enc_4/bias*
_output_shapes
:1*
dtype0

hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1,*&
shared_namehidden_layer_0/kernel

)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
_output_shapes

:1,*
dtype0
~
hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*$
shared_namehidden_layer_0/bias
w
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
_output_shapes
:,*
dtype0

hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,$*&
shared_namehidden_layer_1/kernel

)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
_output_shapes

:,$*
dtype0
~
hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*$
shared_namehidden_layer_1/bias
w
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
_output_shapes
:$*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$
*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:$
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
6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Η5
value½5BΊ5 B³5
έ
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
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Λ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Λ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
Λ

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
Λ

,kernel
-bias
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
¦

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
¦

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
¦

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
* 
j
0
1
2
3
#4
$5
,6
-7
58
69
=10
>11
E12
F13*
.
50
61
=2
>3
E4
F5*
* 
°
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Rserving_default* 
\V
VARIABLE_VALUEenc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

#0
$1*
* 
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

,0
-1*
* 
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 

lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
<
0
1
2
3
#4
$5
,6
-7*
5
0
1
2
3
4
5
6*

v0
w1*
* 
* 
* 

0
1*
* 
* 
* 
* 

0
1*
* 
* 
* 
* 

#0
$1*
* 
* 
* 
* 

,0
-1*
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
* 
* 
* 
* 
* 
8
	xtotal
	ycount
z	variables
{	keras_api*
I
	|total
	}count
~
_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

z	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

|0
}1*

	variables*

serving_default_input_layerPlaceholder*(
_output_shapes
:?????????*
dtype0*
shape:?????????
»
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasenc_4/kernel
enc_4/biashidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_8410576
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp enc_3/kernel/Read/ReadVariableOpenc_3/bias/Read/ReadVariableOp enc_4/kernel/Read/ReadVariableOpenc_4/bias/Read/ReadVariableOp)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
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
 __inference__traced_save_8410793
±
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasenc_4/kernel
enc_4/biashidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/biastotalcounttotal_1count_1*
Tin
2*
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
#__inference__traced_restore_8410857αΦ


τ
B__inference_enc_3_layer_call_and_return_conditional_losses_8410636

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
Υ
κ
/__inference_sequential_50_layer_call_fn_8410435

inputs
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1,
	unknown_8:,
	unknown_9:,$

unknown_10:$

unknown_11:$


unknown_12:

identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410227o
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

ϊ
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045

inputs0
matmul_readvariableop_resource:$
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$
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
:?????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
²
ε
%__inference_signature_wrapper_8410576
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1,
	unknown_8:,
	unknown_9:,$

unknown_10:$

unknown_11:$


unknown_12:

identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_8409925o
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
Υ
κ
/__inference_sequential_50_layer_call_fn_8410402

inputs
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1,
	unknown_8:,
	unknown_9:,$

unknown_10:$

unknown_11:$


unknown_12:

identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410052o
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

φ
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943

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
©H
½

#__inference__traced_restore_8410857
file_prefix1
assignvariableop_enc_1_kernel:
,
assignvariableop_1_enc_1_bias:	3
assignvariableop_2_enc_2_kernel:
Δ,
assignvariableop_3_enc_2_bias:	Δ2
assignvariableop_4_enc_3_kernel:	Δb+
assignvariableop_5_enc_3_bias:b1
assignvariableop_6_enc_4_kernel:b1+
assignvariableop_7_enc_4_bias:1:
(assignvariableop_8_hidden_layer_0_kernel:1,4
&assignvariableop_9_hidden_layer_0_bias:,;
)assignvariableop_10_hidden_layer_1_kernel:,$5
'assignvariableop_11_hidden_layer_1_bias:$9
'assignvariableop_12_output_layer_kernel:$
3
%assignvariableop_13_output_layer_bias:
#
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: 
identity_19’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9η
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ύ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
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
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_enc_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_enc_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp(assignvariableop_8_hidden_layer_0_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp&assignvariableop_9_hidden_layer_0_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_hidden_layer_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp'assignvariableop_11_hidden_layer_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_output_layer_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_output_layer_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ϋ
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: Θ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
₯

φ
B__inference_enc_1_layer_call_and_return_conditional_losses_8410596

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
&
Ν
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410227

inputs!
enc_1_8410191:

enc_1_8410193:	!
enc_2_8410196:
Δ
enc_2_8410198:	Δ 
enc_3_8410201:	Δb
enc_3_8410203:b
enc_4_8410206:b1
enc_4_8410208:1(
hidden_layer_0_8410211:1,$
hidden_layer_0_8410213:,(
hidden_layer_1_8410216:,$$
hidden_layer_1_8410218:$&
output_layer_8410221:$
"
output_layer_8410223:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’enc_4/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’&hidden_layer_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallθ
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_8410191enc_1_8410193*
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
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8410196enc_2_8410198*
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
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8410201enc_3_8410203*
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
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_8410206enc_4_8410208*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0hidden_layer_0_8410211hidden_layer_0_8410213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011΄
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_8410216hidden_layer_1_8410218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0output_layer_8410221output_layer_8410223*
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
Ώ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


σ
B__inference_enc_4_layer_call_and_return_conditional_losses_8410656

inputs0
matmul_readvariableop_resource:b1-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:b1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????1a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????1w
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
Ύ

'__inference_enc_4_layer_call_fn_8410645

inputs
unknown:b1
	unknown_0:1
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????1`
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
δ
ο
/__inference_sequential_50_layer_call_fn_8410083
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1,
	unknown_8:,
	unknown_9:,$

unknown_10:$

unknown_11:$


unknown_12:

identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410052o
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
Π

0__inference_hidden_layer_1_layer_call_fn_8410685

inputs
unknown:,$
	unknown_0:$
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
’

ό
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028

inputs0
matmul_readvariableop_resource:,$-
biasadd_readvariableop_resource:$
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
Μ

.__inference_output_layer_layer_call_fn_8410705

inputs
unknown:$
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045o
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
:?????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
&
?
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410330
input_layer!
enc_1_8410294:

enc_1_8410296:	!
enc_2_8410299:
Δ
enc_2_8410301:	Δ 
enc_3_8410304:	Δb
enc_3_8410306:b
enc_4_8410309:b1
enc_4_8410311:1(
hidden_layer_0_8410314:1,$
hidden_layer_0_8410316:,(
hidden_layer_1_8410319:,$$
hidden_layer_1_8410321:$&
output_layer_8410324:$
"
output_layer_8410326:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’enc_4/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’&hidden_layer_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallν
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_8410294enc_1_8410296*
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
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8410299enc_2_8410301*
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
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8410304enc_3_8410306*
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
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_8410309enc_4_8410311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0hidden_layer_0_8410314hidden_layer_0_8410316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011΄
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_8410319hidden_layer_1_8410321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0output_layer_8410324output_layer_8410326*
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
Ώ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
’

ό
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410696

inputs0
matmul_readvariableop_resource:,$-
biasadd_readvariableop_resource:$
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????$a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
ιN
ί
"__inference__wrapped_model_8409925
input_layerF
2sequential_50_enc_1_matmul_readvariableop_resource:
B
3sequential_50_enc_1_biasadd_readvariableop_resource:	F
2sequential_50_enc_2_matmul_readvariableop_resource:
ΔB
3sequential_50_enc_2_biasadd_readvariableop_resource:	ΔE
2sequential_50_enc_3_matmul_readvariableop_resource:	ΔbA
3sequential_50_enc_3_biasadd_readvariableop_resource:bD
2sequential_50_enc_4_matmul_readvariableop_resource:b1A
3sequential_50_enc_4_biasadd_readvariableop_resource:1M
;sequential_50_hidden_layer_0_matmul_readvariableop_resource:1,J
<sequential_50_hidden_layer_0_biasadd_readvariableop_resource:,M
;sequential_50_hidden_layer_1_matmul_readvariableop_resource:,$J
<sequential_50_hidden_layer_1_biasadd_readvariableop_resource:$K
9sequential_50_output_layer_matmul_readvariableop_resource:$
H
:sequential_50_output_layer_biasadd_readvariableop_resource:

identity’*sequential_50/enc_1/BiasAdd/ReadVariableOp’)sequential_50/enc_1/MatMul/ReadVariableOp’*sequential_50/enc_2/BiasAdd/ReadVariableOp’)sequential_50/enc_2/MatMul/ReadVariableOp’*sequential_50/enc_3/BiasAdd/ReadVariableOp’)sequential_50/enc_3/MatMul/ReadVariableOp’*sequential_50/enc_4/BiasAdd/ReadVariableOp’)sequential_50/enc_4/MatMul/ReadVariableOp’3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp’2sequential_50/hidden_layer_0/MatMul/ReadVariableOp’3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp’2sequential_50/hidden_layer_1/MatMul/ReadVariableOp’1sequential_50/output_layer/BiasAdd/ReadVariableOp’0sequential_50/output_layer/MatMul/ReadVariableOp
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
:?????????b
)sequential_50/enc_4/MatMul/ReadVariableOpReadVariableOp2sequential_50_enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0±
sequential_50/enc_4/MatMulMatMul&sequential_50/enc_3/Relu:activations:01sequential_50/enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1
*sequential_50/enc_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_50_enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0²
sequential_50/enc_4/BiasAddBiasAdd$sequential_50/enc_4/MatMul:product:02sequential_50/enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1x
sequential_50/enc_4/ReluRelu$sequential_50/enc_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????1?
2sequential_50/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp;sequential_50_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:1,*
dtype0Γ
#sequential_50/hidden_layer_0/MatMulMatMul&sequential_50/enc_4/Relu:activations:0:sequential_50/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,¬
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp<sequential_50_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0Ν
$sequential_50/hidden_layer_0/BiasAddBiasAdd-sequential_50/hidden_layer_0/MatMul:product:0;sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,
!sequential_50/hidden_layer_0/SeluSelu-sequential_50/hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????,?
2sequential_50/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp;sequential_50_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:,$*
dtype0Μ
#sequential_50/hidden_layer_1/MatMulMatMul/sequential_50/hidden_layer_0/Selu:activations:0:sequential_50/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$¬
3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp<sequential_50_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0Ν
$sequential_50/hidden_layer_1/BiasAddBiasAdd-sequential_50/hidden_layer_1/MatMul:product:0;sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$
!sequential_50/hidden_layer_1/SeluSelu-sequential_50/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$ͺ
0sequential_50/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_50_output_layer_matmul_readvariableop_resource*
_output_shapes

:$
*
dtype0Θ
!sequential_50/output_layer/MatMulMatMul/sequential_50/hidden_layer_1/Selu:activations:08sequential_50/output_layer/MatMul/ReadVariableOp:value:0*
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
η
NoOpNoOp+^sequential_50/enc_1/BiasAdd/ReadVariableOp*^sequential_50/enc_1/MatMul/ReadVariableOp+^sequential_50/enc_2/BiasAdd/ReadVariableOp*^sequential_50/enc_2/MatMul/ReadVariableOp+^sequential_50/enc_3/BiasAdd/ReadVariableOp*^sequential_50/enc_3/MatMul/ReadVariableOp+^sequential_50/enc_4/BiasAdd/ReadVariableOp*^sequential_50/enc_4/MatMul/ReadVariableOp4^sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp3^sequential_50/hidden_layer_0/MatMul/ReadVariableOp4^sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp3^sequential_50/hidden_layer_1/MatMul/ReadVariableOp2^sequential_50/output_layer/BiasAdd/ReadVariableOp1^sequential_50/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2X
*sequential_50/enc_1/BiasAdd/ReadVariableOp*sequential_50/enc_1/BiasAdd/ReadVariableOp2V
)sequential_50/enc_1/MatMul/ReadVariableOp)sequential_50/enc_1/MatMul/ReadVariableOp2X
*sequential_50/enc_2/BiasAdd/ReadVariableOp*sequential_50/enc_2/BiasAdd/ReadVariableOp2V
)sequential_50/enc_2/MatMul/ReadVariableOp)sequential_50/enc_2/MatMul/ReadVariableOp2X
*sequential_50/enc_3/BiasAdd/ReadVariableOp*sequential_50/enc_3/BiasAdd/ReadVariableOp2V
)sequential_50/enc_3/MatMul/ReadVariableOp)sequential_50/enc_3/MatMul/ReadVariableOp2X
*sequential_50/enc_4/BiasAdd/ReadVariableOp*sequential_50/enc_4/BiasAdd/ReadVariableOp2V
)sequential_50/enc_4/MatMul/ReadVariableOp)sequential_50/enc_4/MatMul/ReadVariableOp2j
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp2h
2sequential_50/hidden_layer_0/MatMul/ReadVariableOp2sequential_50/hidden_layer_0/MatMul/ReadVariableOp2j
3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp2h
2sequential_50/hidden_layer_1/MatMul/ReadVariableOp2sequential_50/hidden_layer_1/MatMul/ReadVariableOp2f
1sequential_50/output_layer/BiasAdd/ReadVariableOp1sequential_50/output_layer/BiasAdd/ReadVariableOp2d
0sequential_50/output_layer/MatMul/ReadVariableOp0sequential_50/output_layer/MatMul/ReadVariableOp:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
’

ό
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011

inputs0
matmul_readvariableop_resource:1,-
biasadd_readvariableop_resource:,
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????,a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
&
?
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410369
input_layer!
enc_1_8410333:

enc_1_8410335:	!
enc_2_8410338:
Δ
enc_2_8410340:	Δ 
enc_3_8410343:	Δb
enc_3_8410345:b
enc_4_8410348:b1
enc_4_8410350:1(
hidden_layer_0_8410353:1,$
hidden_layer_0_8410355:,(
hidden_layer_1_8410358:,$$
hidden_layer_1_8410360:$&
output_layer_8410363:$
"
output_layer_8410365:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’enc_4/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’&hidden_layer_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallν
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_8410333enc_1_8410335*
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
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8410338enc_2_8410340*
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
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8410343enc_3_8410345*
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
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_8410348enc_4_8410350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0hidden_layer_0_8410353hidden_layer_0_8410355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011΄
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_8410358hidden_layer_1_8410360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0output_layer_8410363output_layer_8410365*
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
Ώ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
’

ό
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410676

inputs0
matmul_readvariableop_resource:1,-
biasadd_readvariableop_resource:,
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????,a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
>
ϊ

J__inference_sequential_50_layer_call_and_return_conditional_losses_8410488

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Δ4
%enc_2_biasadd_readvariableop_resource:	Δ7
$enc_3_matmul_readvariableop_resource:	Δb3
%enc_3_biasadd_readvariableop_resource:b6
$enc_4_matmul_readvariableop_resource:b13
%enc_4_biasadd_readvariableop_resource:1?
-hidden_layer_0_matmul_readvariableop_resource:1,<
.hidden_layer_0_biasadd_readvariableop_resource:,?
-hidden_layer_1_matmul_readvariableop_resource:,$<
.hidden_layer_1_biasadd_readvariableop_resource:$=
+output_layer_matmul_readvariableop_resource:$
:
,output_layer_biasadd_readvariableop_resource:

identity’enc_1/BiasAdd/ReadVariableOp’enc_1/MatMul/ReadVariableOp’enc_2/BiasAdd/ReadVariableOp’enc_2/MatMul/ReadVariableOp’enc_3/BiasAdd/ReadVariableOp’enc_3/MatMul/ReadVariableOp’enc_4/BiasAdd/ReadVariableOp’enc_4/MatMul/ReadVariableOp’%hidden_layer_0/BiasAdd/ReadVariableOp’$hidden_layer_0/MatMul/ReadVariableOp’%hidden_layer_1/BiasAdd/ReadVariableOp’$hidden_layer_1/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
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
:?????????b
enc_4/MatMul/ReadVariableOpReadVariableOp$enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0
enc_4/MatMulMatMulenc_3/Relu:activations:0#enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1~
enc_4/BiasAdd/ReadVariableOpReadVariableOp%enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
enc_4/BiasAddBiasAddenc_4/MatMul:product:0$enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1\

enc_4/ReluReluenc_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????1
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:1,*
dtype0
hidden_layer_0/MatMulMatMulenc_4/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,n
hidden_layer_0/SeluSeluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????,
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:,$*
dtype0’
hidden_layer_1/MatMulMatMul!hidden_layer_0/Selu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$n
hidden_layer_1/SeluSeluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:$
*
dtype0
output_layer/MatMulMatMul!hidden_layer_1/Selu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
£
NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp^enc_4/BiasAdd/ReadVariableOp^enc_4/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2<
enc_4/BiasAdd/ReadVariableOpenc_4/BiasAdd/ReadVariableOp2:
enc_4/MatMul/ReadVariableOpenc_4/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


σ
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994

inputs0
matmul_readvariableop_resource:b1-
biasadd_readvariableop_resource:1
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:b1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????1a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????1w
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

φ
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960

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
'__inference_enc_3_layer_call_fn_8410625

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
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977o
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
Α+
£
 __inference__traced_save_8410793
file_prefix+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop+
'savev2_enc_3_kernel_read_readvariableop)
%savev2_enc_3_bias_read_readvariableop+
'savev2_enc_4_kernel_read_readvariableop)
%savev2_enc_4_bias_read_readvariableop4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop2
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
: δ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B °
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop'savev2_enc_3_kernel_read_readvariableop%savev2_enc_3_bias_read_readvariableop'savev2_enc_4_kernel_read_readvariableop%savev2_enc_4_bias_read_readvariableop0savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2
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

identity_1Identity_1:output:0*
_input_shapes
: :
::
Δ:Δ:	Δb:b:b1:1:1,:,:,$:$:$
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

:b1: 

_output_shapes
:1:$	 

_output_shapes

:1,: 


_output_shapes
:,:$ 

_output_shapes

:,$: 

_output_shapes
:$:$ 

_output_shapes

:$
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ε

'__inference_enc_1_layer_call_fn_8410585

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
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943p
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
Ε

'__inference_enc_2_layer_call_fn_8410605

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
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960p
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


τ
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977

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
0__inference_hidden_layer_0_layer_call_fn_8410665

inputs
unknown:1,
	unknown_0:,
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
&
Ν
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410052

inputs!
enc_1_8409944:

enc_1_8409946:	!
enc_2_8409961:
Δ
enc_2_8409963:	Δ 
enc_3_8409978:	Δb
enc_3_8409980:b
enc_4_8409995:b1
enc_4_8409997:1(
hidden_layer_0_8410012:1,$
hidden_layer_0_8410014:,(
hidden_layer_1_8410029:,$$
hidden_layer_1_8410031:$&
output_layer_8410046:$
"
output_layer_8410048:

identity’enc_1/StatefulPartitionedCall’enc_2/StatefulPartitionedCall’enc_3/StatefulPartitionedCall’enc_4/StatefulPartitionedCall’&hidden_layer_0/StatefulPartitionedCall’&hidden_layer_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallθ
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_8409944enc_1_8409946*
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
B__inference_enc_1_layer_call_and_return_conditional_losses_8409943
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_8409961enc_2_8409963*
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
B__inference_enc_2_layer_call_and_return_conditional_losses_8409960
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_8409978enc_3_8409980*
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
B__inference_enc_3_layer_call_and_return_conditional_losses_8409977
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_8409995enc_4_8409997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_enc_4_layer_call_and_return_conditional_losses_8409994«
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0hidden_layer_0_8410012hidden_layer_0_8410014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410011΄
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_8410029hidden_layer_1_8410031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410028¬
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0output_layer_8410046output_layer_8410048*
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410045|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
Ώ
NoOpNoOp^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
δ
ο
/__inference_sequential_50_layer_call_fn_8410291
input_layer
unknown:

	unknown_0:	
	unknown_1:
Δ
	unknown_2:	Δ
	unknown_3:	Δb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1,
	unknown_8:,
	unknown_9:,$

unknown_10:$

unknown_11:$


unknown_12:

identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410227o
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????
%
_user_specified_nameinput_layer
₯

ϊ
I__inference_output_layer_layer_call_and_return_conditional_losses_8410716

inputs0
matmul_readvariableop_resource:$
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$
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
:?????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????$
 
_user_specified_nameinputs
>
ϊ

J__inference_sequential_50_layer_call_and_return_conditional_losses_8410541

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Δ4
%enc_2_biasadd_readvariableop_resource:	Δ7
$enc_3_matmul_readvariableop_resource:	Δb3
%enc_3_biasadd_readvariableop_resource:b6
$enc_4_matmul_readvariableop_resource:b13
%enc_4_biasadd_readvariableop_resource:1?
-hidden_layer_0_matmul_readvariableop_resource:1,<
.hidden_layer_0_biasadd_readvariableop_resource:,?
-hidden_layer_1_matmul_readvariableop_resource:,$<
.hidden_layer_1_biasadd_readvariableop_resource:$=
+output_layer_matmul_readvariableop_resource:$
:
,output_layer_biasadd_readvariableop_resource:

identity’enc_1/BiasAdd/ReadVariableOp’enc_1/MatMul/ReadVariableOp’enc_2/BiasAdd/ReadVariableOp’enc_2/MatMul/ReadVariableOp’enc_3/BiasAdd/ReadVariableOp’enc_3/MatMul/ReadVariableOp’enc_4/BiasAdd/ReadVariableOp’enc_4/MatMul/ReadVariableOp’%hidden_layer_0/BiasAdd/ReadVariableOp’$hidden_layer_0/MatMul/ReadVariableOp’%hidden_layer_1/BiasAdd/ReadVariableOp’$hidden_layer_1/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
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
:?????????b
enc_4/MatMul/ReadVariableOpReadVariableOp$enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0
enc_4/MatMulMatMulenc_3/Relu:activations:0#enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1~
enc_4/BiasAdd/ReadVariableOpReadVariableOp%enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
enc_4/BiasAddBiasAddenc_4/MatMul:product:0$enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1\

enc_4/ReluReluenc_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????1
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:1,*
dtype0
hidden_layer_0/MatMulMatMulenc_4/Relu:activations:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,n
hidden_layer_0/SeluSeluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????,
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:,$*
dtype0’
hidden_layer_1/MatMulMatMul!hidden_layer_0/Selu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$n
hidden_layer_1/SeluSeluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:$
*
dtype0
output_layer/MatMulMatMul!hidden_layer_1/Selu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
£
NoOpNoOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp^enc_4/BiasAdd/ReadVariableOp^enc_4/MatMul/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????: : : : : : : : : : : : : : 2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2<
enc_4/BiasAdd/ReadVariableOpenc_4/BiasAdd/ReadVariableOp2:
enc_4/MatMul/ReadVariableOpenc_4/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

φ
B__inference_enc_2_layer_call_and_return_conditional_losses_8410616

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
tensorflow/serving/predict:μ
χ
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
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
ΰ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ΰ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
ΰ

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
ΰ

,kernel
-bias
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
»

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
»

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer

0
1
2
3
#4
$5
,6
-7
58
69
=10
>11
E12
F13"
trackable_list_wrapper
J
50
61
=2
>3
E4
F5"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_50_layer_call_fn_8410083
/__inference_sequential_50_layer_call_fn_8410402
/__inference_sequential_50_layer_call_fn_8410435
/__inference_sequential_50_layer_call_fn_8410291ΐ
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
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410488
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410541
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410330
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410369ΐ
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
"__inference__wrapped_model_8409925input_layer"
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
Rserving_default"
signature_map
 :
2enc_1/kernel
:2
enc_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_1_layer_call_fn_8410585’
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
B__inference_enc_1_layer_call_and_return_conditional_losses_8410596’
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_2_layer_call_fn_8410605’
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
B__inference_enc_2_layer_call_and_return_conditional_losses_8410616’
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
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_3_layer_call_fn_8410625’
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
B__inference_enc_3_layer_call_and_return_conditional_losses_8410636’
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
:b12enc_4/kernel
:12
enc_4/bias
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_enc_4_layer_call_fn_8410645’
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
B__inference_enc_4_layer_call_and_return_conditional_losses_8410656’
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
':%1,2hidden_layer_0/kernel
!:,2hidden_layer_0/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_hidden_layer_0_layer_call_fn_8410665’
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410676’
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
':%,$2hidden_layer_1/kernel
!:$2hidden_layer_1/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_hidden_layer_1_layer_call_fn_8410685’
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
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410696’
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
%:#$
2output_layer/kernel
:
2output_layer/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_output_layer_layer_call_fn_8410705’
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8410716’
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
X
0
1
2
3
#4
$5
,6
-7"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΠBΝ
%__inference_signature_wrapper_8410576input_layer"
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
0
1"
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
0
1"
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
#0
$1"
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
,0
-1"
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
	xtotal
	ycount
z	variables
{	keras_api"
_tf_keras_metric
_
	|total
	}count
~
_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
-
	variables"
_generic_user_object«
"__inference__wrapped_model_8409925#$,-56=>EF5’2
+’(
&#
input_layer?????????
ͺ ";ͺ8
6
output_layer&#
output_layer?????????
€
B__inference_enc_1_layer_call_and_return_conditional_losses_8410596^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 |
'__inference_enc_1_layer_call_fn_8410585Q0’-
&’#
!
inputs?????????
ͺ "?????????€
B__inference_enc_2_layer_call_and_return_conditional_losses_8410616^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????Δ
 |
'__inference_enc_2_layer_call_fn_8410605Q0’-
&’#
!
inputs?????????
ͺ "?????????Δ£
B__inference_enc_3_layer_call_and_return_conditional_losses_8410636]#$0’-
&’#
!
inputs?????????Δ
ͺ "%’"

0?????????b
 {
'__inference_enc_3_layer_call_fn_8410625P#$0’-
&’#
!
inputs?????????Δ
ͺ "?????????b’
B__inference_enc_4_layer_call_and_return_conditional_losses_8410656\,-/’,
%’"
 
inputs?????????b
ͺ "%’"

0?????????1
 z
'__inference_enc_4_layer_call_fn_8410645O,-/’,
%’"
 
inputs?????????b
ͺ "?????????1«
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8410676\56/’,
%’"
 
inputs?????????1
ͺ "%’"

0?????????,
 
0__inference_hidden_layer_0_layer_call_fn_8410665O56/’,
%’"
 
inputs?????????1
ͺ "?????????,«
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8410696\=>/’,
%’"
 
inputs?????????,
ͺ "%’"

0?????????$
 
0__inference_hidden_layer_1_layer_call_fn_8410685O=>/’,
%’"
 
inputs?????????,
ͺ "?????????$©
I__inference_output_layer_layer_call_and_return_conditional_losses_8410716\EF/’,
%’"
 
inputs?????????$
ͺ "%’"

0?????????

 
.__inference_output_layer_layer_call_fn_8410705OEF/’,
%’"
 
inputs?????????$
ͺ "?????????
Δ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410330v#$,-56=>EF=’:
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
 Δ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410369v#$,-56=>EF=’:
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
 Ώ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410488q#$,-56=>EF8’5
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
 Ώ
J__inference_sequential_50_layer_call_and_return_conditional_losses_8410541q#$,-56=>EF8’5
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
 
/__inference_sequential_50_layer_call_fn_8410083i#$,-56=>EF=’:
3’0
&#
input_layer?????????
p 

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8410291i#$,-56=>EF=’:
3’0
&#
input_layer?????????
p

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8410402d#$,-56=>EF8’5
.’+
!
inputs?????????
p 

 
ͺ "?????????

/__inference_sequential_50_layer_call_fn_8410435d#$,-56=>EF8’5
.’+
!
inputs?????????
p

 
ͺ "?????????
½
%__inference_signature_wrapper_8410576#$,-56=>EFD’A
’ 
:ͺ7
5
input_layer&#
input_layer?????????";ͺ8
6
output_layer&#
output_layer?????????
