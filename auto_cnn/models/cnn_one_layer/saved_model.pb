
Ú«
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68à

convolutional_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*+
shared_nameconvolutional_layer/kernel

.convolutional_layer/kernel/Read/ReadVariableOpReadVariableOpconvolutional_layer/kernel*&
_output_shapes
:7*
dtype0

convolutional_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*)
shared_nameconvolutional_layer/bias

,convolutional_layer/bias/Read/ReadVariableOpReadVariableOpconvolutional_layer/bias*
_output_shapes
:7*
dtype0

hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	7·*&
shared_namehidden_layer_0/kernel

)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
_output_shapes
:	7·*
dtype0

hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:·*$
shared_namehidden_layer_0/bias
x
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
_output_shapes	
:·*
dtype0

hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
·Ý*&
shared_namehidden_layer_1/kernel

)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel* 
_output_shapes
:
·Ý*
dtype0

hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ý*$
shared_namehidden_layer_1/bias
x
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
_output_shapes	
:Ý*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Àî
*$
shared_nameoutput_layer/kernel
}
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel* 
_output_shapes
:
Àî
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
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
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
¬-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ç,
valueÝ,BÚ, BÓ,

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
¦

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*

.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
:
<iter
	=decay
>learning_rate
?momentum*
<
0
1
2
3
&4
'5
46
57*
<
0
1
2
3
&4
'5
46
57*
* 
°
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Eserving_default* 
jd
VARIABLE_VALUEconvolutional_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconvolutional_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
e_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

d0
e1*
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
	ftotal
	gcount
h	variables
i	keras_api*
H
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

h	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

j0
k1*

m	variables*

)serving_default_convolutional_layer_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCall)serving_default_convolutional_layer_inputconvolutional_layer/kernelconvolutional_layer/biashidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/bias*
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_7985408
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ø
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.convolutional_layer/kernel/Read/ReadVariableOp,convolutional_layer/bias/Read/ReadVariableOp)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
 __inference__traced_save_7985620
³
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconvolutional_layer/kernelconvolutional_layer/biashidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
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
#__inference__traced_restore_7985678â«
Õ

K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7985478

inputs4
!tensordot_readvariableop_resource:	7·.
biasadd_readvariableop_resource:	·
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	7·*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:·Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:·*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
­

ü
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976

inputs2
matmul_readvariableop_resource:
Àî
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Àî
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
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî
 
_user_specified_nameinputs
ö
ø
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985163
convolutional_layer_input5
convolutional_layer_7985140:7)
convolutional_layer_7985142:7)
hidden_layer_0_7985146:	7·%
hidden_layer_0_7985148:	·*
hidden_layer_1_7985151:
·Ý%
hidden_layer_1_7985153:	Ý(
output_layer_7985157:
Àî
"
output_layer_7985159:

identity¢+convolutional_layer/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallº
+convolutional_layer/StatefulPartitionedCallStatefulPartitionedCallconvolutional_layer_inputconvolutional_layer_7985140convolutional_layer_7985142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876
!max_pooling_layer/PartitionedCallPartitionedCall4convolutional_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855¸
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall*max_pooling_layer/PartitionedCall:output:0hidden_layer_0_7985146hidden_layer_0_7985148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914½
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_7985151hidden_layer_1_7985153*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951è
flatten_50/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#flatten_50/PartitionedCall:output:0output_layer_7985157output_layer_7985159*
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
GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp,^convolutional_layer/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+convolutional_layer/StatefulPartitionedCall+convolutional_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input
é	
Í
/__inference_sequential_50_layer_call_fn_7985214

inputs!
unknown:7
	unknown_0:7
	unknown_1:	7·
	unknown_2:	·
	unknown_3:
·Ý
	unknown_4:	Ý
	unknown_5:
Àî

	unknown_6:

identity¢StatefulPartitionedCall­
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
GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_7984983o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
A
½	
#__inference__traced_restore_7985678
file_prefixE
+assignvariableop_convolutional_layer_kernel:79
+assignvariableop_1_convolutional_layer_bias:7;
(assignvariableop_2_hidden_layer_0_kernel:	7·5
&assignvariableop_3_hidden_layer_0_bias:	·<
(assignvariableop_4_hidden_layer_1_kernel:
·Ý5
&assignvariableop_5_hidden_layer_1_bias:	Ý:
&assignvariableop_6_output_layer_kernel:
Àî
2
$assignvariableop_7_output_layer_bias:
%
assignvariableop_8_sgd_iter:	 &
assignvariableop_9_sgd_decay: /
%assignvariableop_10_sgd_learning_rate: *
 assignvariableop_11_sgd_momentum: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: 
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9×
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B ó
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp+assignvariableop_convolutional_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_convolutional_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_0_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_0_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_1_biasIdentity_5:output:0"/device:CPU:0*
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
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¯
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
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
÷
 
0__inference_hidden_layer_1_layer_call_fn_7985487

inputs
unknown:
·Ý
	unknown_0:	Ý
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ·: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
Ú

K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951

inputs5
!tensordot_readvariableop_resource:
·Ý.
biasadd_readvariableop_resource:	Ý
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
·Ý*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ÝY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ý*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ·: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
é	
Í
/__inference_sequential_50_layer_call_fn_7985235

inputs!
unknown:7
	unknown_0:7
	unknown_1:	7·
	unknown_2:	·
	unknown_3:
·Ý
	unknown_4:	Ý
	unknown_5:
Àî

	unknown_6:

identity¢StatefulPartitionedCall­
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
GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985097o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
å
J__inference_sequential_50_layer_call_and_return_conditional_losses_7984983

inputs5
convolutional_layer_7984877:7)
convolutional_layer_7984879:7)
hidden_layer_0_7984915:	7·%
hidden_layer_0_7984917:	·*
hidden_layer_1_7984952:
·Ý%
hidden_layer_1_7984954:	Ý(
output_layer_7984977:
Àî
"
output_layer_7984979:

identity¢+convolutional_layer/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall§
+convolutional_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconvolutional_layer_7984877convolutional_layer_7984879*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876
!max_pooling_layer/PartitionedCallPartitionedCall4convolutional_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855¸
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall*max_pooling_layer/PartitionedCall:output:0hidden_layer_0_7984915hidden_layer_0_7984917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914½
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_7984952hidden_layer_1_7984954*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951è
flatten_50/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#flatten_50/PartitionedCall:output:0output_layer_7984977output_layer_7984979*
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
GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp,^convolutional_layer/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+convolutional_layer/StatefulPartitionedCall+convolutional_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
O
3__inference_max_pooling_layer_layer_call_fn_7985433

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7985438

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä(
é
 __inference__traced_save_7985620
file_prefix9
5savev2_convolutional_layer_kernel_read_readvariableop7
3savev2_convolutional_layer_bias_read_readvariableop4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
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
: Ô
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B ü
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_convolutional_layer_kernel_read_readvariableop3savev2_convolutional_layer_bias_read_readvariableop0savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
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
c: :7:7:	7·:·:
·Ý:Ý:
Àî
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:7: 

_output_shapes
:7:%!

_output_shapes
:	7·:!

_output_shapes	
:·:&"
 
_output_shapes
:
·Ý:!

_output_shapes	
:Ý:&"
 
_output_shapes
:
Àî
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ã^
È
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985310

inputsL
2convolutional_layer_conv2d_readvariableop_resource:7A
3convolutional_layer_biasadd_readvariableop_resource:7C
0hidden_layer_0_tensordot_readvariableop_resource:	7·=
.hidden_layer_0_biasadd_readvariableop_resource:	·D
0hidden_layer_1_tensordot_readvariableop_resource:
·Ý=
.hidden_layer_1_biasadd_readvariableop_resource:	Ý?
+output_layer_matmul_readvariableop_resource:
Àî
:
,output_layer_biasadd_readvariableop_resource:

identity¢*convolutional_layer/BiasAdd/ReadVariableOp¢)convolutional_layer/Conv2D/ReadVariableOp¢%hidden_layer_0/BiasAdd/ReadVariableOp¢'hidden_layer_0/Tensordot/ReadVariableOp¢%hidden_layer_1/BiasAdd/ReadVariableOp¢'hidden_layer_1/Tensordot/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¤
)convolutional_layer/Conv2D/ReadVariableOpReadVariableOp2convolutional_layer_conv2d_readvariableop_resource*&
_output_shapes
:7*
dtype0Â
convolutional_layer/Conv2DConv2Dinputs1convolutional_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
paddingVALID*
strides

*convolutional_layer/BiasAdd/ReadVariableOpReadVariableOp3convolutional_layer_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¹
convolutional_layer/BiasAddBiasAdd#convolutional_layer/Conv2D:output:02convolutional_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
convolutional_layer/ReluRelu$convolutional_layer/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¹
max_pooling_layer/MaxPoolMaxPool&convolutional_layer/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
ksize
*
paddingVALID*
strides

'hidden_layer_0/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_0_tensordot_readvariableop_resource*
_output_shapes
:	7·*
dtype0g
hidden_layer_0/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
hidden_layer_0/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          p
hidden_layer_0/Tensordot/ShapeShape"max_pooling_layer/MaxPool:output:0*
T0*
_output_shapes
:h
&hidden_layer_0/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!hidden_layer_0/Tensordot/GatherV2GatherV2'hidden_layer_0/Tensordot/Shape:output:0&hidden_layer_0/Tensordot/free:output:0/hidden_layer_0/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(hidden_layer_0/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#hidden_layer_0/Tensordot/GatherV2_1GatherV2'hidden_layer_0/Tensordot/Shape:output:0&hidden_layer_0/Tensordot/axes:output:01hidden_layer_0/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
hidden_layer_0/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
hidden_layer_0/Tensordot/ProdProd*hidden_layer_0/Tensordot/GatherV2:output:0'hidden_layer_0/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 hidden_layer_0/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
hidden_layer_0/Tensordot/Prod_1Prod,hidden_layer_0/Tensordot/GatherV2_1:output:0)hidden_layer_0/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$hidden_layer_0/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
hidden_layer_0/Tensordot/concatConcatV2&hidden_layer_0/Tensordot/free:output:0&hidden_layer_0/Tensordot/axes:output:0-hidden_layer_0/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
hidden_layer_0/Tensordot/stackPack&hidden_layer_0/Tensordot/Prod:output:0(hidden_layer_0/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
"hidden_layer_0/Tensordot/transpose	Transpose"max_pooling_layer/MaxPool:output:0(hidden_layer_0/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7·
 hidden_layer_0/Tensordot/ReshapeReshape&hidden_layer_0/Tensordot/transpose:y:0'hidden_layer_0/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
hidden_layer_0/Tensordot/MatMulMatMul)hidden_layer_0/Tensordot/Reshape:output:0/hidden_layer_0/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·k
 hidden_layer_0/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:·h
&hidden_layer_0/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!hidden_layer_0/Tensordot/concat_1ConcatV2*hidden_layer_0/Tensordot/GatherV2:output:0)hidden_layer_0/Tensordot/Const_2:output:0/hidden_layer_0/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:µ
hidden_layer_0/TensordotReshape)hidden_layer_0/Tensordot/MatMul:product:0*hidden_layer_0/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype0®
hidden_layer_0/BiasAddBiasAdd!hidden_layer_0/Tensordot:output:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·w
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
'hidden_layer_1/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_1_tensordot_readvariableop_resource* 
_output_shapes
:
·Ý*
dtype0g
hidden_layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
hidden_layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          o
hidden_layer_1/Tensordot/ShapeShape!hidden_layer_0/Relu:activations:0*
T0*
_output_shapes
:h
&hidden_layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!hidden_layer_1/Tensordot/GatherV2GatherV2'hidden_layer_1/Tensordot/Shape:output:0&hidden_layer_1/Tensordot/free:output:0/hidden_layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(hidden_layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#hidden_layer_1/Tensordot/GatherV2_1GatherV2'hidden_layer_1/Tensordot/Shape:output:0&hidden_layer_1/Tensordot/axes:output:01hidden_layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
hidden_layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
hidden_layer_1/Tensordot/ProdProd*hidden_layer_1/Tensordot/GatherV2:output:0'hidden_layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 hidden_layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
hidden_layer_1/Tensordot/Prod_1Prod,hidden_layer_1/Tensordot/GatherV2_1:output:0)hidden_layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$hidden_layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
hidden_layer_1/Tensordot/concatConcatV2&hidden_layer_1/Tensordot/free:output:0&hidden_layer_1/Tensordot/axes:output:0-hidden_layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
hidden_layer_1/Tensordot/stackPack&hidden_layer_1/Tensordot/Prod:output:0(hidden_layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
"hidden_layer_1/Tensordot/transpose	Transpose!hidden_layer_0/Relu:activations:0(hidden_layer_1/Tensordot/concat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ··
 hidden_layer_1/Tensordot/ReshapeReshape&hidden_layer_1/Tensordot/transpose:y:0'hidden_layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
hidden_layer_1/Tensordot/MatMulMatMul)hidden_layer_1/Tensordot/Reshape:output:0/hidden_layer_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝk
 hidden_layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Ýh
&hidden_layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!hidden_layer_1/Tensordot/concat_1ConcatV2*hidden_layer_1/Tensordot/GatherV2:output:0)hidden_layer_1/Tensordot/Const_2:output:0/hidden_layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:µ
hidden_layer_1/TensordotReshape)hidden_layer_1/Tensordot/MatMul:product:0*hidden_layer_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:Ý*
dtype0®
hidden_layer_1/BiasAddBiasAdd!hidden_layer_1/Tensordot:output:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝw
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝa
flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@w  
flatten_50/ReshapeReshape!hidden_layer_1/Relu:activations:0flatten_50/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource* 
_output_shapes
:
Àî
*
dtype0
output_layer/MatMulMatMulflatten_50/Reshape:output:0*output_layer/MatMul/ReadVariableOp:value:0*
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

NoOpNoOp+^convolutional_layer/BiasAdd/ReadVariableOp*^convolutional_layer/Conv2D/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp(^hidden_layer_0/Tensordot/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp(^hidden_layer_1/Tensordot/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2X
*convolutional_layer/BiasAdd/ReadVariableOp*convolutional_layer/BiasAdd/ReadVariableOp2V
)convolutional_layer/Conv2D/ReadVariableOp)convolutional_layer/Conv2D/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2R
'hidden_layer_0/Tensordot/ReadVariableOp'hidden_layer_0/Tensordot/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2R
'hidden_layer_1/Tensordot/ReadVariableOp'hidden_layer_1/Tensordot/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
c
G__inference_flatten_50_layer_call_and_return_conditional_losses_7985529

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@w  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀîZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÝ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
 
_user_specified_nameinputs
ð	
Ö
%__inference_signature_wrapper_7985408
convolutional_layer_input!
unknown:7
	unknown_0:7
	unknown_1:	7·
	unknown_2:	·
	unknown_3:
·Ý
	unknown_4:	Ý
	unknown_5:
Àî

	unknown_6:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconvolutional_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
"__inference__wrapped_model_7984846o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input
Í
c
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@w  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀîZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÝ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
 
_user_specified_nameinputs


P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7985428

inputs8
conv2d_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:7*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

.__inference_output_layer_layer_call_fn_7985538

inputs
unknown:
Àî

	unknown_0:

identity¢StatefulPartitionedCallÞ
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
GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976o
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
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀî: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî
 
_user_specified_nameinputs
¹
H
,__inference_flatten_50_layer_call_fn_7985523

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÝ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
 
_user_specified_nameinputs
½
å
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985097

inputs5
convolutional_layer_7985074:7)
convolutional_layer_7985076:7)
hidden_layer_0_7985080:	7·%
hidden_layer_0_7985082:	·*
hidden_layer_1_7985085:
·Ý%
hidden_layer_1_7985087:	Ý(
output_layer_7985091:
Àî
"
output_layer_7985093:

identity¢+convolutional_layer/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall§
+convolutional_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconvolutional_layer_7985074convolutional_layer_7985076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876
!max_pooling_layer/PartitionedCallPartitionedCall4convolutional_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855¸
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall*max_pooling_layer/PartitionedCall:output:0hidden_layer_0_7985080hidden_layer_0_7985082*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914½
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_7985085hidden_layer_1_7985087*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951è
flatten_50/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#flatten_50/PartitionedCall:output:0output_layer_7985091output_layer_7985093*
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
GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp,^convolutional_layer/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+convolutional_layer/StatefulPartitionedCall+convolutional_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

à
/__inference_sequential_50_layer_call_fn_7985137
convolutional_layer_input!
unknown:7
	unknown_0:7
	unknown_1:	7·
	unknown_2:	·
	unknown_3:
·Ý
	unknown_4:	Ý
	unknown_5:
Àî

	unknown_6:

identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallconvolutional_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985097o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input
Ú

K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7985518

inputs5
!tensordot_readvariableop_resource:
·Ý.
biasadd_readvariableop_resource:	Ý
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
·Ý*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ÝY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ý*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ·: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
 
_user_specified_nameinputs
­

ü
I__inference_output_layer_layer_call_and_return_conditional_losses_7985549

inputs2
matmul_readvariableop_resource:
Àî
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Àî
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
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî
 
_user_specified_nameinputs
ô

0__inference_hidden_layer_0_layer_call_fn_7985447

inputs
unknown:	7·
	unknown_0:	·
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ7: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
ö
ø
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985189
convolutional_layer_input5
convolutional_layer_7985166:7)
convolutional_layer_7985168:7)
hidden_layer_0_7985172:	7·%
hidden_layer_0_7985174:	·*
hidden_layer_1_7985177:
·Ý%
hidden_layer_1_7985179:	Ý(
output_layer_7985183:
Àî
"
output_layer_7985185:

identity¢+convolutional_layer/StatefulPartitionedCall¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallº
+convolutional_layer/StatefulPartitionedCallStatefulPartitionedCallconvolutional_layer_inputconvolutional_layer_7985166convolutional_layer_7985168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876
!max_pooling_layer/PartitionedCallPartitionedCall4convolutional_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855¸
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall*max_pooling_layer/PartitionedCall:output:0hidden_layer_0_7985172hidden_layer_0_7985174*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914½
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_7985177hidden_layer_1_7985179*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7984951è
flatten_50/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_50_layer_call_and_return_conditional_losses_7984963 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#flatten_50/PartitionedCall:output:0output_layer_7985183output_layer_7985185*
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
GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_7984976|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp,^convolutional_layer/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+convolutional_layer/StatefulPartitionedCall+convolutional_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input


P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876

inputs8
conv2d_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:7*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ër
	
"__inference__wrapped_model_7984846
convolutional_layer_inputZ
@sequential_50_convolutional_layer_conv2d_readvariableop_resource:7O
Asequential_50_convolutional_layer_biasadd_readvariableop_resource:7Q
>sequential_50_hidden_layer_0_tensordot_readvariableop_resource:	7·K
<sequential_50_hidden_layer_0_biasadd_readvariableop_resource:	·R
>sequential_50_hidden_layer_1_tensordot_readvariableop_resource:
·ÝK
<sequential_50_hidden_layer_1_biasadd_readvariableop_resource:	ÝM
9sequential_50_output_layer_matmul_readvariableop_resource:
Àî
H
:sequential_50_output_layer_biasadd_readvariableop_resource:

identity¢8sequential_50/convolutional_layer/BiasAdd/ReadVariableOp¢7sequential_50/convolutional_layer/Conv2D/ReadVariableOp¢3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp¢5sequential_50/hidden_layer_0/Tensordot/ReadVariableOp¢3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp¢5sequential_50/hidden_layer_1/Tensordot/ReadVariableOp¢1sequential_50/output_layer/BiasAdd/ReadVariableOp¢0sequential_50/output_layer/MatMul/ReadVariableOpÀ
7sequential_50/convolutional_layer/Conv2D/ReadVariableOpReadVariableOp@sequential_50_convolutional_layer_conv2d_readvariableop_resource*&
_output_shapes
:7*
dtype0ñ
(sequential_50/convolutional_layer/Conv2DConv2Dconvolutional_layer_input?sequential_50/convolutional_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
paddingVALID*
strides
¶
8sequential_50/convolutional_layer/BiasAdd/ReadVariableOpReadVariableOpAsequential_50_convolutional_layer_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0ã
)sequential_50/convolutional_layer/BiasAddBiasAdd1sequential_50/convolutional_layer/Conv2D:output:0@sequential_50/convolutional_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
&sequential_50/convolutional_layer/ReluRelu2sequential_50/convolutional_layer/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Õ
'sequential_50/max_pooling_layer/MaxPoolMaxPool4sequential_50/convolutional_layer/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
ksize
*
paddingVALID*
strides
µ
5sequential_50/hidden_layer_0/Tensordot/ReadVariableOpReadVariableOp>sequential_50_hidden_layer_0_tensordot_readvariableop_resource*
_output_shapes
:	7·*
dtype0u
+sequential_50/hidden_layer_0/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
+sequential_50/hidden_layer_0/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          
,sequential_50/hidden_layer_0/Tensordot/ShapeShape0sequential_50/max_pooling_layer/MaxPool:output:0*
T0*
_output_shapes
:v
4sequential_50/hidden_layer_0/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/sequential_50/hidden_layer_0/Tensordot/GatherV2GatherV25sequential_50/hidden_layer_0/Tensordot/Shape:output:04sequential_50/hidden_layer_0/Tensordot/free:output:0=sequential_50/hidden_layer_0/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6sequential_50/hidden_layer_0/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1sequential_50/hidden_layer_0/Tensordot/GatherV2_1GatherV25sequential_50/hidden_layer_0/Tensordot/Shape:output:04sequential_50/hidden_layer_0/Tensordot/axes:output:0?sequential_50/hidden_layer_0/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,sequential_50/hidden_layer_0/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+sequential_50/hidden_layer_0/Tensordot/ProdProd8sequential_50/hidden_layer_0/Tensordot/GatherV2:output:05sequential_50/hidden_layer_0/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.sequential_50/hidden_layer_0/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-sequential_50/hidden_layer_0/Tensordot/Prod_1Prod:sequential_50/hidden_layer_0/Tensordot/GatherV2_1:output:07sequential_50/hidden_layer_0/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2sequential_50/hidden_layer_0/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-sequential_50/hidden_layer_0/Tensordot/concatConcatV24sequential_50/hidden_layer_0/Tensordot/free:output:04sequential_50/hidden_layer_0/Tensordot/axes:output:0;sequential_50/hidden_layer_0/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,sequential_50/hidden_layer_0/Tensordot/stackPack4sequential_50/hidden_layer_0/Tensordot/Prod:output:06sequential_50/hidden_layer_0/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:á
0sequential_50/hidden_layer_0/Tensordot/transpose	Transpose0sequential_50/max_pooling_layer/MaxPool:output:06sequential_50/hidden_layer_0/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7á
.sequential_50/hidden_layer_0/Tensordot/ReshapeReshape4sequential_50/hidden_layer_0/Tensordot/transpose:y:05sequential_50/hidden_layer_0/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
-sequential_50/hidden_layer_0/Tensordot/MatMulMatMul7sequential_50/hidden_layer_0/Tensordot/Reshape:output:0=sequential_50/hidden_layer_0/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·y
.sequential_50/hidden_layer_0/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:·v
4sequential_50/hidden_layer_0/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/sequential_50/hidden_layer_0/Tensordot/concat_1ConcatV28sequential_50/hidden_layer_0/Tensordot/GatherV2:output:07sequential_50/hidden_layer_0/Tensordot/Const_2:output:0=sequential_50/hidden_layer_0/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ß
&sequential_50/hidden_layer_0/TensordotReshape7sequential_50/hidden_layer_0/Tensordot/MatMul:product:08sequential_50/hidden_layer_0/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·­
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp<sequential_50_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype0Ø
$sequential_50/hidden_layer_0/BiasAddBiasAdd/sequential_50/hidden_layer_0/Tensordot:output:0;sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!sequential_50/hidden_layer_0/ReluRelu-sequential_50/hidden_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·¶
5sequential_50/hidden_layer_1/Tensordot/ReadVariableOpReadVariableOp>sequential_50_hidden_layer_1_tensordot_readvariableop_resource* 
_output_shapes
:
·Ý*
dtype0u
+sequential_50/hidden_layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
+sequential_50/hidden_layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          
,sequential_50/hidden_layer_1/Tensordot/ShapeShape/sequential_50/hidden_layer_0/Relu:activations:0*
T0*
_output_shapes
:v
4sequential_50/hidden_layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/sequential_50/hidden_layer_1/Tensordot/GatherV2GatherV25sequential_50/hidden_layer_1/Tensordot/Shape:output:04sequential_50/hidden_layer_1/Tensordot/free:output:0=sequential_50/hidden_layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6sequential_50/hidden_layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1sequential_50/hidden_layer_1/Tensordot/GatherV2_1GatherV25sequential_50/hidden_layer_1/Tensordot/Shape:output:04sequential_50/hidden_layer_1/Tensordot/axes:output:0?sequential_50/hidden_layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,sequential_50/hidden_layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+sequential_50/hidden_layer_1/Tensordot/ProdProd8sequential_50/hidden_layer_1/Tensordot/GatherV2:output:05sequential_50/hidden_layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.sequential_50/hidden_layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-sequential_50/hidden_layer_1/Tensordot/Prod_1Prod:sequential_50/hidden_layer_1/Tensordot/GatherV2_1:output:07sequential_50/hidden_layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2sequential_50/hidden_layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-sequential_50/hidden_layer_1/Tensordot/concatConcatV24sequential_50/hidden_layer_1/Tensordot/free:output:04sequential_50/hidden_layer_1/Tensordot/axes:output:0;sequential_50/hidden_layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,sequential_50/hidden_layer_1/Tensordot/stackPack4sequential_50/hidden_layer_1/Tensordot/Prod:output:06sequential_50/hidden_layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:á
0sequential_50/hidden_layer_1/Tensordot/transpose	Transpose/sequential_50/hidden_layer_0/Relu:activations:06sequential_50/hidden_layer_1/Tensordot/concat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·á
.sequential_50/hidden_layer_1/Tensordot/ReshapeReshape4sequential_50/hidden_layer_1/Tensordot/transpose:y:05sequential_50/hidden_layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
-sequential_50/hidden_layer_1/Tensordot/MatMulMatMul7sequential_50/hidden_layer_1/Tensordot/Reshape:output:0=sequential_50/hidden_layer_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝy
.sequential_50/hidden_layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Ýv
4sequential_50/hidden_layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/sequential_50/hidden_layer_1/Tensordot/concat_1ConcatV28sequential_50/hidden_layer_1/Tensordot/GatherV2:output:07sequential_50/hidden_layer_1/Tensordot/Const_2:output:0=sequential_50/hidden_layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ß
&sequential_50/hidden_layer_1/TensordotReshape7sequential_50/hidden_layer_1/Tensordot/MatMul:product:08sequential_50/hidden_layer_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ­
3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp<sequential_50_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:Ý*
dtype0Ø
$sequential_50/hidden_layer_1/BiasAddBiasAdd/sequential_50/hidden_layer_1/Tensordot:output:0;sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
!sequential_50/hidden_layer_1/ReluRelu-sequential_50/hidden_layer_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝo
sequential_50/flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@w  ¹
 sequential_50/flatten_50/ReshapeReshape/sequential_50/hidden_layer_1/Relu:activations:0'sequential_50/flatten_50/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî¬
0sequential_50/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_50_output_layer_matmul_readvariableop_resource* 
_output_shapes
:
Àî
*
dtype0Â
!sequential_50/output_layer/MatMulMatMul)sequential_50/flatten_50/Reshape:output:08sequential_50/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
1sequential_50/output_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_50_output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ç
"sequential_50/output_layer/BiasAddBiasAdd+sequential_50/output_layer/MatMul:product:09sequential_50/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"sequential_50/output_layer/SoftmaxSoftmax+sequential_50/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
IdentityIdentity,sequential_50/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
NoOpNoOp9^sequential_50/convolutional_layer/BiasAdd/ReadVariableOp8^sequential_50/convolutional_layer/Conv2D/ReadVariableOp4^sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp6^sequential_50/hidden_layer_0/Tensordot/ReadVariableOp4^sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp6^sequential_50/hidden_layer_1/Tensordot/ReadVariableOp2^sequential_50/output_layer/BiasAdd/ReadVariableOp1^sequential_50/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8sequential_50/convolutional_layer/BiasAdd/ReadVariableOp8sequential_50/convolutional_layer/BiasAdd/ReadVariableOp2r
7sequential_50/convolutional_layer/Conv2D/ReadVariableOp7sequential_50/convolutional_layer/Conv2D/ReadVariableOp2j
3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp3sequential_50/hidden_layer_0/BiasAdd/ReadVariableOp2n
5sequential_50/hidden_layer_0/Tensordot/ReadVariableOp5sequential_50/hidden_layer_0/Tensordot/ReadVariableOp2j
3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp3sequential_50/hidden_layer_1/BiasAdd/ReadVariableOp2n
5sequential_50/hidden_layer_1/Tensordot/ReadVariableOp5sequential_50/hidden_layer_1/Tensordot/ReadVariableOp2f
1sequential_50/output_layer/BiasAdd/ReadVariableOp1sequential_50/output_layer/BiasAdd/ReadVariableOp2d
0sequential_50/output_layer/MatMul/ReadVariableOp0sequential_50/output_layer/MatMul/ReadVariableOp:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input

j
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7984855

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7984914

inputs4
!tensordot_readvariableop_resource:	7·.
biasadd_readvariableop_resource:	·
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	7·*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:·Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:·*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs

ª
5__inference_convolutional_layer_layer_call_fn_7985417

inputs!
unknown:7
	unknown_0:7
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7984876w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã^
È
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985385

inputsL
2convolutional_layer_conv2d_readvariableop_resource:7A
3convolutional_layer_biasadd_readvariableop_resource:7C
0hidden_layer_0_tensordot_readvariableop_resource:	7·=
.hidden_layer_0_biasadd_readvariableop_resource:	·D
0hidden_layer_1_tensordot_readvariableop_resource:
·Ý=
.hidden_layer_1_biasadd_readvariableop_resource:	Ý?
+output_layer_matmul_readvariableop_resource:
Àî
:
,output_layer_biasadd_readvariableop_resource:

identity¢*convolutional_layer/BiasAdd/ReadVariableOp¢)convolutional_layer/Conv2D/ReadVariableOp¢%hidden_layer_0/BiasAdd/ReadVariableOp¢'hidden_layer_0/Tensordot/ReadVariableOp¢%hidden_layer_1/BiasAdd/ReadVariableOp¢'hidden_layer_1/Tensordot/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¤
)convolutional_layer/Conv2D/ReadVariableOpReadVariableOp2convolutional_layer_conv2d_readvariableop_resource*&
_output_shapes
:7*
dtype0Â
convolutional_layer/Conv2DConv2Dinputs1convolutional_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
paddingVALID*
strides

*convolutional_layer/BiasAdd/ReadVariableOpReadVariableOp3convolutional_layer_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¹
convolutional_layer/BiasAddBiasAdd#convolutional_layer/Conv2D:output:02convolutional_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
convolutional_layer/ReluRelu$convolutional_layer/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¹
max_pooling_layer/MaxPoolMaxPool&convolutional_layer/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
ksize
*
paddingVALID*
strides

'hidden_layer_0/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_0_tensordot_readvariableop_resource*
_output_shapes
:	7·*
dtype0g
hidden_layer_0/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
hidden_layer_0/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          p
hidden_layer_0/Tensordot/ShapeShape"max_pooling_layer/MaxPool:output:0*
T0*
_output_shapes
:h
&hidden_layer_0/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!hidden_layer_0/Tensordot/GatherV2GatherV2'hidden_layer_0/Tensordot/Shape:output:0&hidden_layer_0/Tensordot/free:output:0/hidden_layer_0/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(hidden_layer_0/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#hidden_layer_0/Tensordot/GatherV2_1GatherV2'hidden_layer_0/Tensordot/Shape:output:0&hidden_layer_0/Tensordot/axes:output:01hidden_layer_0/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
hidden_layer_0/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
hidden_layer_0/Tensordot/ProdProd*hidden_layer_0/Tensordot/GatherV2:output:0'hidden_layer_0/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 hidden_layer_0/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
hidden_layer_0/Tensordot/Prod_1Prod,hidden_layer_0/Tensordot/GatherV2_1:output:0)hidden_layer_0/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$hidden_layer_0/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
hidden_layer_0/Tensordot/concatConcatV2&hidden_layer_0/Tensordot/free:output:0&hidden_layer_0/Tensordot/axes:output:0-hidden_layer_0/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
hidden_layer_0/Tensordot/stackPack&hidden_layer_0/Tensordot/Prod:output:0(hidden_layer_0/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
"hidden_layer_0/Tensordot/transpose	Transpose"max_pooling_layer/MaxPool:output:0(hidden_layer_0/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7·
 hidden_layer_0/Tensordot/ReshapeReshape&hidden_layer_0/Tensordot/transpose:y:0'hidden_layer_0/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
hidden_layer_0/Tensordot/MatMulMatMul)hidden_layer_0/Tensordot/Reshape:output:0/hidden_layer_0/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·k
 hidden_layer_0/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:·h
&hidden_layer_0/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!hidden_layer_0/Tensordot/concat_1ConcatV2*hidden_layer_0/Tensordot/GatherV2:output:0)hidden_layer_0/Tensordot/Const_2:output:0/hidden_layer_0/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:µ
hidden_layer_0/TensordotReshape)hidden_layer_0/Tensordot/MatMul:product:0*hidden_layer_0/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:·*
dtype0®
hidden_layer_0/BiasAddBiasAdd!hidden_layer_0/Tensordot:output:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·w
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
'hidden_layer_1/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_1_tensordot_readvariableop_resource* 
_output_shapes
:
·Ý*
dtype0g
hidden_layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
hidden_layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          o
hidden_layer_1/Tensordot/ShapeShape!hidden_layer_0/Relu:activations:0*
T0*
_output_shapes
:h
&hidden_layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!hidden_layer_1/Tensordot/GatherV2GatherV2'hidden_layer_1/Tensordot/Shape:output:0&hidden_layer_1/Tensordot/free:output:0/hidden_layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(hidden_layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#hidden_layer_1/Tensordot/GatherV2_1GatherV2'hidden_layer_1/Tensordot/Shape:output:0&hidden_layer_1/Tensordot/axes:output:01hidden_layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
hidden_layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
hidden_layer_1/Tensordot/ProdProd*hidden_layer_1/Tensordot/GatherV2:output:0'hidden_layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 hidden_layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
hidden_layer_1/Tensordot/Prod_1Prod,hidden_layer_1/Tensordot/GatherV2_1:output:0)hidden_layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$hidden_layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
hidden_layer_1/Tensordot/concatConcatV2&hidden_layer_1/Tensordot/free:output:0&hidden_layer_1/Tensordot/axes:output:0-hidden_layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
hidden_layer_1/Tensordot/stackPack&hidden_layer_1/Tensordot/Prod:output:0(hidden_layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
"hidden_layer_1/Tensordot/transpose	Transpose!hidden_layer_0/Relu:activations:0(hidden_layer_1/Tensordot/concat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ··
 hidden_layer_1/Tensordot/ReshapeReshape&hidden_layer_1/Tensordot/transpose:y:0'hidden_layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
hidden_layer_1/Tensordot/MatMulMatMul)hidden_layer_1/Tensordot/Reshape:output:0/hidden_layer_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝk
 hidden_layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Ýh
&hidden_layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!hidden_layer_1/Tensordot/concat_1ConcatV2*hidden_layer_1/Tensordot/GatherV2:output:0)hidden_layer_1/Tensordot/Const_2:output:0/hidden_layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:µ
hidden_layer_1/TensordotReshape)hidden_layer_1/Tensordot/MatMul:product:0*hidden_layer_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:Ý*
dtype0®
hidden_layer_1/BiasAddBiasAdd!hidden_layer_1/Tensordot:output:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝw
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝa
flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@w  
flatten_50/ReshapeReshape!hidden_layer_1/Relu:activations:0flatten_50/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀî
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource* 
_output_shapes
:
Àî
*
dtype0
output_layer/MatMulMatMulflatten_50/Reshape:output:0*output_layer/MatMul/ReadVariableOp:value:0*
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

NoOpNoOp+^convolutional_layer/BiasAdd/ReadVariableOp*^convolutional_layer/Conv2D/ReadVariableOp&^hidden_layer_0/BiasAdd/ReadVariableOp(^hidden_layer_0/Tensordot/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp(^hidden_layer_1/Tensordot/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2X
*convolutional_layer/BiasAdd/ReadVariableOp*convolutional_layer/BiasAdd/ReadVariableOp2V
)convolutional_layer/Conv2D/ReadVariableOp)convolutional_layer/Conv2D/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2R
'hidden_layer_0/Tensordot/ReadVariableOp'hidden_layer_0/Tensordot/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2R
'hidden_layer_1/Tensordot/ReadVariableOp'hidden_layer_1/Tensordot/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

à
/__inference_sequential_50_layer_call_fn_7985002
convolutional_layer_input!
unknown:7
	unknown_0:7
	unknown_1:	7·
	unknown_2:	·
	unknown_3:
·Ý
	unknown_4:	Ý
	unknown_5:
Àî

	unknown_6:

identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallconvolutional_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_50_layer_call_and_return_conditional_losses_7984983o
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
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_nameconvolutional_layer_input"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Û
serving_defaultÇ
g
convolutional_layer_inputJ
+serving_default_convolutional_layer_input:0ÿÿÿÿÿÿÿÿÿ@
output_layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:¹u

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
»

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
I
<iter
	=decay
>learning_rate
?momentum"
	optimizer
X
0
1
2
3
&4
'5
46
57"
trackable_list_wrapper
X
0
1
2
3
&4
'5
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_50_layer_call_fn_7985002
/__inference_sequential_50_layer_call_fn_7985214
/__inference_sequential_50_layer_call_fn_7985235
/__inference_sequential_50_layer_call_fn_7985137À
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
ö2ó
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985310
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985385
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985163
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985189À
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
ßBÜ
"__inference__wrapped_model_7984846convolutional_layer_input"
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
Eserving_default"
signature_map
4:272convolutional_layer/kernel
&:$72convolutional_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_convolutional_layer_layer_call_fn_7985417¢
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
ú2÷
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7985428¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_max_pooling_layer_layer_call_fn_7985433¢
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
ø2õ
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7985438¢
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
(:&	7·2hidden_layer_0/kernel
": ·2hidden_layer_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_hidden_layer_0_layer_call_fn_7985447¢
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
õ2ò
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7985478¢
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
·Ý2hidden_layer_1/kernel
": Ý2hidden_layer_1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_hidden_layer_1_layer_call_fn_7985487¢
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
õ2ò
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7985518¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_50_layer_call_fn_7985523¢
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
ñ2î
G__inference_flatten_50_layer_call_and_return_conditional_losses_7985529¢
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
':%
Àî
2output_layer/kernel
:
2output_layer/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_output_layer_layer_call_fn_7985538¢
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
ó2ð
I__inference_output_layer_layer_call_and_return_conditional_losses_7985549¢
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
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
%__inference_signature_wrapper_7985408convolutional_layer_input"
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
	ftotal
	gcount
h	variables
i	keras_api"
_tf_keras_metric
^
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_objectº
"__inference__wrapped_model_7984846&'45J¢G
@¢=
;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ
À
P__inference_convolutional_layer_layer_call_and_return_conditional_losses_7985428l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ7
 
5__inference_convolutional_layer_layer_call_fn_7985417_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ7®
G__inference_flatten_50_layer_call_and_return_conditional_losses_7985529c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÝ
ª "'¢$

0ÿÿÿÿÿÿÿÿÿÀî
 
,__inference_flatten_50_layer_call_fn_7985523V8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÝ
ª "ÿÿÿÿÿÿÿÿÿÀî¼
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_7985478m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ7
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ·
 
0__inference_hidden_layer_0_layer_call_fn_7985447`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ7
ª "!ÿÿÿÿÿÿÿÿÿ·½
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_7985518n&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ·
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÝ
 
0__inference_hidden_layer_1_layer_call_fn_7985487a&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ·
ª "!ÿÿÿÿÿÿÿÿÿÝñ
N__inference_max_pooling_layer_layer_call_and_return_conditional_losses_7985438R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling_layer_layer_call_fn_7985433R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
I__inference_output_layer_layer_call_and_return_conditional_losses_7985549^451¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿÀî
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_output_layer_layer_call_fn_7985538Q451¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿÀî
ª "ÿÿÿÿÿÿÿÿÿ
Ô
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985163&'45R¢O
H¢E
;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ô
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985189&'45R¢O
H¢E
;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 À
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985310r&'45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 À
J__inference_sequential_50_layer_call_and_return_conditional_losses_7985385r&'45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 «
/__inference_sequential_50_layer_call_fn_7985002x&'45R¢O
H¢E
;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
«
/__inference_sequential_50_layer_call_fn_7985137x&'45R¢O
H¢E
;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

/__inference_sequential_50_layer_call_fn_7985214e&'45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

/__inference_sequential_50_layer_call_fn_7985235e&'45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ú
%__inference_signature_wrapper_7985408°&'45g¢d
¢ 
]ªZ
X
convolutional_layer_input;8
convolutional_layer_inputÿÿÿÿÿÿÿÿÿ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ
