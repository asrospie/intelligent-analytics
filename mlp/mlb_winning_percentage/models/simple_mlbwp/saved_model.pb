??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
z
hidden_2/kernelVarHandleOp* 
shared_namehidden_2/kernel*
shape
:*
dtype0*
_output_shapes
: 
s
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel*
_output_shapes

:*
dtype0
r
hidden_2/biasVarHandleOp*
shape:*
_output_shapes
: *
shared_namehidden_2/bias*
dtype0
k
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
dtype0*
_output_shapes
:
z
output_2/kernelVarHandleOp* 
shared_nameoutput_2/kernel*
dtype0*
shape
:*
_output_shapes
: 
s
#output_2/kernel/Read/ReadVariableOpReadVariableOpoutput_2/kernel*
dtype0*
_output_shapes

:
r
output_2/biasVarHandleOp*
_output_shapes
: *
shared_nameoutput_2/bias*
dtype0*
shape:
k
!output_2/bias/Read/ReadVariableOpReadVariableOpoutput_2/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
dtype0	*
shape: *
_output_shapes
: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
dtype0*
shared_name	SGD/decay*
_output_shapes
: *
shape: 
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
dtype0*
_output_shapes
: 
v
SGD/learning_rateVarHandleOp*"
shared_nameSGD/learning_rate*
shape: *
_output_shapes
: *
dtype0
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
dtype0*
_output_shapes
: 
l
SGD/momentumVarHandleOp*
_output_shapes
: *
shape: *
dtype0*
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
shared_nametotal*
shape: *
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
shared_name	count_1*
shape: *
dtype0*
_output_shapes
: 
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
dtype0*
_output_shapes
: 

NoOpNoOp
?
ConstConst"/device:CPU:0*
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
R

	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
6
iter
	decay
learning_rate
momentum

0
1
2
3
 

0
1
2
3
?
non_trainable_variables
	variables
metrics

 layers
regularization_losses
trainable_variables
!layer_regularization_losses
 
 
 
 
?
"non_trainable_variables

	variables
#metrics

$layers
regularization_losses
trainable_variables
%layer_regularization_losses
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
&non_trainable_variables
	variables
'metrics

(layers
regularization_losses
trainable_variables
)layer_regularization_losses
[Y
VARIABLE_VALUEoutput_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
*non_trainable_variables
	variables
+metrics

,layers
regularization_losses
trainable_variables
-layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	0total
	1count
2
_fn_kwargs
3	variables
4regularization_losses
5trainable_variables
6	keras_api
x
	7total
	8count
9
_fn_kwargs
:	variables
;regularization_losses
<trainable_variables
=	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

00
11
 
 
?
>non_trainable_variables
3	variables
?metrics

@layers
4regularization_losses
5trainable_variables
Alayer_regularization_losses
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

70
81
 
 
?
Bnon_trainable_variables
:	variables
Cmetrics

Dlayers
;regularization_losses
<trainable_variables
Elayer_regularization_losses

00
11
 
 
 

70
81
 
 
 *
_output_shapes
: 
x
serving_default_inputPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputhidden_2/kernelhidden_2/biasoutput_2/kerneloutput_2/bias*,
f'R%
#__inference_signature_wrapper_15787*
Tout
2*
Tin	
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15907**
config_proto

CPU

GPU 2J 8
O
saver_filenamePlaceholder*
dtype0*
shape: *
_output_shapes
: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#output_2/kernel/Read/ReadVariableOp!output_2/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15941*'
f"R 
__inference__traced_save_15940*
_output_shapes
: *
Tout
2*
Tin
2	
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_2/kernelhidden_2/biasoutput_2/kerneloutput_2/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1**
f%R#
!__inference__traced_restore_15989*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15990*
_output_shapes
: ԟ
?
?
,__inference_sequential_2_layer_call_fn_15834

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*,
_gradient_op_typePartitionedCall-15744*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_15743*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15807

inputs)
%hidden_matmul_readvariableop_resource*
&hidden_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??hidden/BiasAdd/ReadVariableOp?hidden/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0w
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0^
hidden/ReluReluhidden/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
output/MatMulMatMulhidden/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
output/SigmoidSigmoidoutput/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityoutput/Sigmoid:y:0^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
?"
?
__inference__traced_save_15940
file_prefix.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop.
*savev2_output_2_kernel_read_readvariableop,
(savev2_output_2_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_c8a32a3dc8794ea2a415362af4932e4c/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*+
value"B B B B B B B B B B B B B *
dtype0?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_output_2_kernel_read_readvariableop(savev2_output_2_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"/device:CPU:0*
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*G
_input_shapes6
4: ::::: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : 
?	
?
A__inference_hidden_layer_call_and_return_conditional_losses_15672

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15743

inputs)
%hidden_statefulpartitionedcall_args_1)
%hidden_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??hidden/StatefulPartitionedCall?output/StatefulPartitionedCall?
hidden/StatefulPartitionedCallStatefulPartitionedCallinputs%hidden_statefulpartitionedcall_args_1%hidden_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15678*J
fERC
A__inference_hidden_layer_call_and_return_conditional_losses_15672*
Tin
2?
output/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_15700*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15706*
Tin
2**
config_proto

CPU

GPU 2J 8?
IdentityIdentity'output/StatefulPartitionedCall:output:0^hidden/StatefulPartitionedCall^output/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
 __inference__wrapped_model_15655	
input6
2sequential_2_hidden_matmul_readvariableop_resource7
3sequential_2_hidden_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??*sequential_2/hidden/BiasAdd/ReadVariableOp?)sequential_2/hidden/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?
)sequential_2/hidden/MatMul/ReadVariableOpReadVariableOp2sequential_2_hidden_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
sequential_2/hidden/MatMulMatMulinput1sequential_2/hidden/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
*sequential_2/hidden/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_hidden_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential_2/hidden/BiasAddBiasAdd$sequential_2/hidden/MatMul:product:02sequential_2/hidden/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0x
sequential_2/hidden/ReluRelu$sequential_2/hidden/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
sequential_2/output/MatMulMatMul&sequential_2/hidden/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0~
sequential_2/output/SigmoidSigmoid$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitysequential_2/output/Sigmoid:y:0+^sequential_2/hidden/BiasAdd/ReadVariableOp*^sequential_2/hidden/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2X
*sequential_2/hidden/BiasAdd/ReadVariableOp*sequential_2/hidden/BiasAdd/ReadVariableOp2V
)sequential_2/hidden/MatMul/ReadVariableOp)sequential_2/hidden/MatMul/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp: :% !

_user_specified_nameinput: : : 
?
?
,__inference_sequential_2_layer_call_fn_15751	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15744*
Tout
2*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_15743*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :% !

_user_specified_nameinput: : : 
?1
?
!__inference__traced_restore_15989
file_prefix$
 assignvariableop_hidden_2_kernel$
 assignvariableop_1_hidden_2_bias&
"assignvariableop_2_output_2_kernel$
 assignvariableop_3_output_2_bias
assignvariableop_4_sgd_iter 
assignvariableop_5_sgd_decay(
$assignvariableop_6_sgd_learning_rate#
assignvariableop_7_sgd_momentum
assignvariableop_8_total
assignvariableop_9_count
assignvariableop_10_total_1
assignvariableop_11_count_1
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*+
value"B B B B B B B B B B B B B *
dtype0?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2	*D
_output_shapes2
0::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:|
AssignVariableOpAssignVariableOp assignvariableop_hidden_2_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_2_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_output_2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp assignvariableop_3_output_2_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:{
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0|
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0x
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0}
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0}
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_1: :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
?	
?
A__inference_output_layer_call_and_return_conditional_losses_15700

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
&__inference_output_layer_call_fn_15879

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15706*
Tout
2*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_15700*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
A__inference_output_layer_call_and_return_conditional_losses_15872

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
,__inference_sequential_2_layer_call_fn_15773	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-15766*
Tin	
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_15765*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameinput: : : : 
?
?
,__inference_sequential_2_layer_call_fn_15843

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_15765*,
_gradient_op_typePartitionedCall-15766*'
_output_shapes
:?????????*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
&__inference_hidden_layer_call_fn_15861

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*J
fERC
A__inference_hidden_layer_call_and_return_conditional_losses_15672*'
_output_shapes
:?????????*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15678?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15718	
input)
%hidden_statefulpartitionedcall_args_1)
%hidden_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??hidden/StatefulPartitionedCall?output/StatefulPartitionedCall?
hidden/StatefulPartitionedCallStatefulPartitionedCallinput%hidden_statefulpartitionedcall_args_1%hidden_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tout
2*'
_output_shapes
:?????????*J
fERC
A__inference_hidden_layer_call_and_return_conditional_losses_15672*,
_gradient_op_typePartitionedCall-15678*
Tin
2?
output/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15706*
Tin
2*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_15700*'
_output_shapes
:??????????
IdentityIdentity'output/StatefulPartitionedCall:output:0^hidden/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:% !

_user_specified_nameinput: : : : 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15765

inputs)
%hidden_statefulpartitionedcall_args_1)
%hidden_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??hidden/StatefulPartitionedCall?output/StatefulPartitionedCall?
hidden/StatefulPartitionedCallStatefulPartitionedCallinputs%hidden_statefulpartitionedcall_args_1%hidden_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*J
fERC
A__inference_hidden_layer_call_and_return_conditional_losses_15672**
config_proto

CPU

GPU 2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-15678?
output/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_15700*
Tin
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-15706*
Tout
2?
IdentityIdentity'output/StatefulPartitionedCall:output:0^hidden/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15825

inputs)
%hidden_matmul_readvariableop_resource*
&hidden_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??hidden/BiasAdd/ReadVariableOp?hidden/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:w
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0^
hidden/ReluReluhidden/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
output/MatMulMatMulhidden/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
output/SigmoidSigmoidoutput/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityoutput/Sigmoid:y:0^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?
?
#__inference_signature_wrapper_15787	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*'
_output_shapes
:?????????*
Tout
2*
Tin	
2*)
f$R"
 __inference__wrapped_model_15655*,
_gradient_op_typePartitionedCall-15780**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :% !

_user_specified_nameinput: : : 
?	
?
A__inference_hidden_layer_call_and_return_conditional_losses_15854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15730	
input)
%hidden_statefulpartitionedcall_args_1)
%hidden_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??hidden/StatefulPartitionedCall?output/StatefulPartitionedCall?
hidden/StatefulPartitionedCallStatefulPartitionedCallinput%hidden_statefulpartitionedcall_args_1%hidden_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-15678*J
fERC
A__inference_hidden_layer_call_and_return_conditional_losses_15672?
output/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-15706*
Tout
2*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_15700**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:??????????
IdentityIdentity'output/StatefulPartitionedCall:output:0^hidden/StatefulPartitionedCall^output/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:% !

_user_specified_nameinput: : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?w
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
F__call__
*G&call_and_return_all_conditional_losses
H_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "hidden", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 13]}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "hidden", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 13]}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error", "mean_absolute_error"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?

	variables
regularization_losses
trainable_variables
	keras_api
I__call__
*J&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 13], "config": {"batch_input_shape": [null, 13], "dtype": "float32", "sparse": false, "name": "input"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}
I
iter
	decay
learning_rate
momentum"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
non_trainable_variables
	variables
metrics

 layers
regularization_losses
trainable_variables
!layer_regularization_losses
F__call__
H_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"non_trainable_variables

	variables
#metrics

$layers
regularization_losses
trainable_variables
%layer_regularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
!:2hidden_2/kernel
:2hidden_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
&non_trainable_variables
	variables
'metrics

(layers
regularization_losses
trainable_variables
)layer_regularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
!:2output_2/kernel
:2output_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
*non_trainable_variables
	variables
+metrics

,layers
regularization_losses
trainable_variables
-layer_regularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	0total
	1count
2
_fn_kwargs
3	variables
4regularization_losses
5trainable_variables
6	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
?
	7total
	8count
9
_fn_kwargs
:	variables
;regularization_losses
<trainable_variables
=	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mean_absolute_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_absolute_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables
3	variables
?metrics

@layers
4regularization_losses
5trainable_variables
Alayer_regularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables
:	variables
Cmetrics

Dlayers
;regularization_losses
<trainable_variables
Elayer_regularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
,__inference_sequential_2_layer_call_fn_15751
,__inference_sequential_2_layer_call_fn_15773
,__inference_sequential_2_layer_call_fn_15834
,__inference_sequential_2_layer_call_fn_15843?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15730
G__inference_sequential_2_layer_call_and_return_conditional_losses_15718
G__inference_sequential_2_layer_call_and_return_conditional_losses_15825
G__inference_sequential_2_layer_call_and_return_conditional_losses_15807?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_15655?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input?????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
&__inference_hidden_layer_call_fn_15861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_hidden_layer_call_and_return_conditional_losses_15854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_output_layer_call_fn_15879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_output_layer_call_and_return_conditional_losses_15872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0B.
#__inference_signature_wrapper_15787input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
A__inference_hidden_layer_call_and_return_conditional_losses_15854\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_15834Y7?4
-?*
 ?
inputs?????????
p

 
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_15825f7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15807f7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_15773X6?3
,?)
?
input?????????
p 

 
? "??????????y
&__inference_output_layer_call_fn_15879O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_output_layer_call_and_return_conditional_losses_15872\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
#__inference_signature_wrapper_15787p7?4
? 
-?*
(
input?
input?????????"/?,
*
output ?
output??????????
 __inference__wrapped_model_15655g.?+
$?!
?
input?????????
? "/?,
*
output ?
output??????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_15730e6?3
,?)
?
input?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_15843Y7?4
-?*
 ?
inputs?????????
p 

 
? "??????????y
&__inference_hidden_layer_call_fn_15861O/?,
%?"
 ?
inputs?????????
? "???????????
,__inference_sequential_2_layer_call_fn_15751X6?3
,?)
?
input?????????
p

 
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_15718e6?3
,?)
?
input?????????
p

 
? "%?"
?
0?????????
? 