??
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
shapeshape?"serve*2.0.02unknown8??
?
hidden_layer_0/kernelVarHandleOp*
shape:
??*
_output_shapes
: *&
shared_namehidden_layer_0/kernel*
dtype0
?
)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
dtype0* 
_output_shapes
:
??

hidden_layer_0/biasVarHandleOp*$
shared_namehidden_layer_0/bias*
shape:?*
dtype0*
_output_shapes
: 
x
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
dtype0*
_output_shapes	
:?
?
output_layer/kernelVarHandleOp*
_output_shapes
: *$
shared_nameoutput_layer/kernel*
dtype0*
shape:	?

|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	?
*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:
*
dtype0
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
shape: *
_output_shapes
: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
hidden_layer_0/kernel/mVarHandleOp*
dtype0*(
shared_namehidden_layer_0/kernel/m*
_output_shapes
: *
shape:
??
?
+hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/m*
dtype0* 
_output_shapes
:
??
?
hidden_layer_0/bias/mVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *&
shared_namehidden_layer_0/bias/m
|
)hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/m*
_output_shapes	
:?*
dtype0
?
output_layer/kernel/mVarHandleOp*&
shared_nameoutput_layer/kernel/m*
dtype0*
_output_shapes
: *
shape:	?

?
)output_layer/kernel/m/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/m*
_output_shapes
:	?
*
dtype0
~
output_layer/bias/mVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*$
shared_nameoutput_layer/bias/m
w
'output_layer/bias/m/Read/ReadVariableOpReadVariableOpoutput_layer/bias/m*
dtype0*
_output_shapes
:

?
hidden_layer_0/kernel/vVarHandleOp*
shape:
??*(
shared_namehidden_layer_0/kernel/v*
_output_shapes
: *
dtype0
?
+hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/v* 
_output_shapes
:
??*
dtype0
?
hidden_layer_0/bias/vVarHandleOp*&
shared_namehidden_layer_0/bias/v*
_output_shapes
: *
shape:?*
dtype0
|
)hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/v*
_output_shapes	
:?*
dtype0
?
output_layer/kernel/vVarHandleOp*&
shared_nameoutput_layer/kernel/v*
_output_shapes
: *
shape:	?
*
dtype0
?
)output_layer/kernel/v/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/v*
dtype0*
_output_shapes
:	?

~
output_layer/bias/vVarHandleOp*$
shared_nameoutput_layer/bias/v*
_output_shapes
: *
dtype0*
shape:

w
'output_layer/bias/v/Read/ReadVariableOpReadVariableOpoutput_layer/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
?
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *?
value?B? B?
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Hm?m@mAmBvCvDvEvF

0
1
2
3
 

0
1
2
3
?
	variables
regularization_losses

layers
trainable_variables
 non_trainable_variables
!layer_regularization_losses
"metrics
 
 
 
 
?
	variables
regularization_losses

#layers
trainable_variables
$metrics
%layer_regularization_losses
&non_trainable_variables
 
 
 
?
	variables
regularization_losses

'layers
trainable_variables
(metrics
)layer_regularization_losses
*non_trainable_variables
a_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses

+layers
trainable_variables
,metrics
-layer_regularization_losses
.non_trainable_variables
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses

/layers
trainable_variables
0metrics
1layer_regularization_losses
2non_trainable_variables

0
1
2
 
 

30
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
 
 
 
x
	4total
	5count
6
_fn_kwargs
7	variables
8regularization_losses
9trainable_variables
:	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
 
 
?
7	variables
8regularization_losses

;layers
9trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
 
 
 

40
51
}
VARIABLE_VALUEhidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_input_layer_inputPlaceholder*+
_output_shapes
:?????????* 
shape:?????????*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputhidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/bias*'
_output_shapes
:?????????
**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_8643602*
Tin	
2*.
_gradient_op_typePartitionedCall-8643741*
Tout
2
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+hidden_layer_0/kernel/m/Read/ReadVariableOp)hidden_layer_0/bias/m/Read/ReadVariableOp)output_layer/kernel/m/Read/ReadVariableOp'output_layer/bias/m/Read/ReadVariableOp+hidden_layer_0/kernel/v/Read/ReadVariableOp)hidden_layer_0/bias/v/Read/ReadVariableOp)output_layer/kernel/v/Read/ReadVariableOp'output_layer/bias/v/Read/ReadVariableOpConst**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *
Tout
2*)
f$R"
 __inference__traced_save_8643776*.
_gradient_op_typePartitionedCall-8643777
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0/kernelhidden_layer_0/biasoutput_layer/kerneloutput_layer/biastotalcounthidden_layer_0/kernel/mhidden_layer_0/bias/moutput_layer/kernel/moutput_layer/bias/mhidden_layer_0/kernel/vhidden_layer_0/bias/voutput_layer/kernel/voutput_layer/bias/v*,
f'R%
#__inference__traced_restore_8643831*
Tout
2*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8643832*
Tin
2??
?
I
-__inference_input_layer_layer_call_fn_8643673

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8643469*(
_output_shapes
:??????????*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_8643702

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?<
?
#__inference__traced_restore_8643831
file_prefix*
&assignvariableop_hidden_layer_0_kernel*
&assignvariableop_1_hidden_layer_0_bias*
&assignvariableop_2_output_layer_kernel(
$assignvariableop_3_output_layer_bias
assignvariableop_4_total
assignvariableop_5_count.
*assignvariableop_6_hidden_layer_0_kernel_m,
(assignvariableop_7_hidden_layer_0_bias_m,
(assignvariableop_8_output_layer_kernel_m*
&assignvariableop_9_output_layer_bias_m/
+assignvariableop_10_hidden_layer_0_kernel_v-
)assignvariableop_11_hidden_layer_0_bias_v-
)assignvariableop_12_output_layer_kernel_v+
'assignvariableop_13_output_layer_bias_v
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0?
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:x
AssignVariableOp_4AssignVariableOpassignvariableop_4_totalIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0x
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_hidden_layer_0_kernel_mIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_hidden_layer_0_bias_mIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_output_layer_kernel_mIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_output_layer_bias_mIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_hidden_layer_0_kernel_vIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_hidden_layer_0_bias_vIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_output_layer_kernel_vIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_output_layer_bias_vIdentity_13:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : 
?
?
,__inference_sequential_layer_call_fn_8643568
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*'
_output_shapes
:?????????
*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8643560**
config_proto

GPU 

CPU2J 8*
Tin	
2*.
_gradient_op_typePartitionedCall-8643561?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463

inputs
identity^
Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643624

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
output_layer/MatMulMatMul!hidden_layer_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?(
?
 __inference__traced_save_8643776
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_hidden_layer_0_kernel_m_read_readvariableop4
0savev2_hidden_layer_0_bias_m_read_readvariableop4
0savev2_output_layer_kernel_m_read_readvariableop2
.savev2_output_layer_bias_m_read_readvariableop6
2savev2_hidden_layer_0_kernel_v_read_readvariableop4
0savev2_hidden_layer_0_bias_v_read_readvariableop4
0savev2_output_layer_kernel_v_read_readvariableop2
.savev2_output_layer_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*<
value3B1 B+_temp_183bb886a95b41789aefa0064a04b7e5/part*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*/
value&B$B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_hidden_layer_0_kernel_m_read_readvariableop0savev2_hidden_layer_0_bias_m_read_readvariableop0savev2_output_layer_kernel_m_read_readvariableop.savev2_output_layer_bias_m_read_readvariableop2savev2_hidden_layer_0_kernel_v_read_readvariableop0savev2_hidden_layer_0_bias_v_read_readvariableop0savev2_output_layer_kernel_v_read_readvariableop.savev2_output_layer_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*?
_input_shapesv
t: :
??:?:	?
:
: : :
??:?:	?
:
:
??:?:	?
:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_8643668

inputs
identity^
Reshape/shapeConst*
valueB"????  *
_output_shapes
:*
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643583

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-8643469*(
_output_shapes
:??????????*
Tin
2**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463*
Tout
2?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tout
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487*
Tin
2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643493**
config_proto

GPU 

CPU2J 8?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tin
2*.
_gradient_op_typePartitionedCall-8643521**
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515*
Tout
2*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643644

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????  }
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
output_layer/MatMulMatMul!hidden_layer_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
?
?
,__inference_sequential_layer_call_fn_8643662

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*
Tout
2*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-8643584*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8643583?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643684

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
%__inference_signature_wrapper_8643602
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*'
_output_shapes
:?????????
*
Tout
2**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_8643451*.
_gradient_op_typePartitionedCall-8643595*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : 
?
?
"__inference__wrapped_model_8643451
input_layer_input<
8sequential_hidden_layer_0_matmul_readvariableop_resource=
9sequential_hidden_layer_0_biasadd_readvariableop_resource:
6sequential_output_layer_matmul_readvariableop_resource;
7sequential_output_layer_biasadd_readvariableop_resource
identity??0sequential/hidden_layer_0/BiasAdd/ReadVariableOp?/sequential/hidden_layer_0/MatMul/ReadVariableOp?.sequential/output_layer/BiasAdd/ReadVariableOp?-sequential/output_layer/MatMul/ReadVariableOpu
$sequential/input_layer/Reshape/shapeConst*
dtype0*
valueB"????  *
_output_shapes
:?
sequential/input_layer/ReshapeReshapeinput_layer_input-sequential/input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
/sequential/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
 sequential/hidden_layer_0/MatMulMatMul'sequential/input_layer/Reshape:output:07sequential/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
!sequential/hidden_layer_0/BiasAddBiasAdd*sequential/hidden_layer_0/MatMul:product:08sequential/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hidden_layer_0/ReluRelu*sequential/hidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
sequential/output_layer/MatMulMatMul,sequential/hidden_layer_0/Relu:activations:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
sequential/output_layer/SoftmaxSoftmax(sequential/output_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentity)sequential/output_layer/Softmax:softmax:01^sequential/hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/hidden_layer_0/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2d
0sequential/hidden_layer_0/BiasAdd/ReadVariableOp0sequential/hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_0/MatMul/ReadVariableOp/sequential/hidden_layer_0/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp:1 -
+
_user_specified_nameinput_layer_input: : : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643533
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463*.
_gradient_op_typePartitionedCall-8643469*(
_output_shapes
:??????????*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643493**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????
*
Tout
2**
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515*.
_gradient_op_typePartitionedCall-8643521?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall: :1 -
+
_user_specified_nameinput_layer_input: : : 
?
?
.__inference_output_layer_layer_call_fn_8643709

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????
*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515*
Tin
2*.
_gradient_op_typePartitionedCall-8643521**
config_proto

GPU 

CPU2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643546
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*.
_gradient_op_typePartitionedCall-8643469**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463*
Tout
2?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487*
Tin
2*.
_gradient_op_typePartitionedCall-8643493*(
_output_shapes
:??????????*
Tout
2**
config_proto

GPU 

CPU2J 8?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-8643521*
Tin
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tout
2?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : 
?
?
,__inference_sequential_layer_call_fn_8643653

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-8643561**
config_proto

GPU 

CPU2J 8*
Tout
2*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8643560*'
_output_shapes
:?????????
*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643560

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-8643469*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:??????????*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643463?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487*
Tin
2*.
_gradient_op_typePartitionedCall-8643493*(
_output_shapes
:???????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8643515*
Tout
2*.
_gradient_op_typePartitionedCall-8643521*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?
?
,__inference_sequential_layer_call_fn_8643591
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8643583*'
_output_shapes
:?????????
*
Tin	
2*.
_gradient_op_typePartitionedCall-8643584**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :1 -
+
_user_specified_nameinput_layer_input: : : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
0__inference_hidden_layer_0_layer_call_fn_8643691

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643493*
Tin
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643487?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
input_layer_input>
#serving_default_input_layer_input:0?????????@
output_layer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
G_default_save_signature
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*N&call_and_return_all_conditional_losses
O__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 704}}}}
[m?m@mAmBvCvDvEvF"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
	variables
regularization_losses

layers
trainable_variables
 non_trainable_variables
!layer_regularization_losses
"metrics
I__call__
G_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses

#layers
trainable_variables
$metrics
%layer_regularization_losses
&non_trainable_variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses

'layers
trainable_variables
(metrics
)layer_regularization_losses
*non_trainable_variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_0/kernel
": ?2hidden_layer_0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses

+layers
trainable_variables
,metrics
-layer_regularization_losses
.non_trainable_variables
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
&:$	?
2output_layer/kernel
:
2output_layer/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses

/layers
trainable_variables
0metrics
1layer_regularization_losses
2non_trainable_variables
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
30"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	4total
	5count
6
_fn_kwargs
7	variables
8regularization_losses
9trainable_variables
:	keras_api
*S&call_and_return_all_conditional_losses
T__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables
8regularization_losses

;layers
9trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
):'
??2hidden_layer_0/kernel/m
": ?2hidden_layer_0/bias/m
&:$	?
2output_layer/kernel/m
:
2output_layer/bias/m
):'
??2hidden_layer_0/kernel/v
": ?2hidden_layer_0/bias/v
&:$	?
2output_layer/kernel/v
:
2output_layer/bias/v
?2?
"__inference__wrapped_model_8643451?
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
annotations? *4?1
/?,
input_layer_input?????????
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_8643546
G__inference_sequential_layer_call_and_return_conditional_losses_8643533
G__inference_sequential_layer_call_and_return_conditional_losses_8643624
G__inference_sequential_layer_call_and_return_conditional_losses_8643644?
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
?2?
,__inference_sequential_layer_call_fn_8643662
,__inference_sequential_layer_call_fn_8643568
,__inference_sequential_layer_call_fn_8643591
,__inference_sequential_layer_call_fn_8643653?
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
H__inference_input_layer_layer_call_and_return_conditional_losses_8643668?
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
-__inference_input_layer_layer_call_fn_8643673?
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643684?
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
0__inference_hidden_layer_0_layer_call_fn_8643691?
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
I__inference_output_layer_layer_call_and_return_conditional_losses_8643702?
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
.__inference_output_layer_layer_call_fn_8643709?
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
>B<
%__inference_signature_wrapper_8643602input_layer_input
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
H__inference_input_layer_layer_call_and_return_conditional_losses_8643668]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
I__inference_output_layer_layer_call_and_return_conditional_losses_8643702]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
.__inference_output_layer_layer_call_fn_8643709P0?-
&?#
!?
inputs??????????
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643533uF?C
<?9
/?,
input_layer_input?????????
p

 
? "%?"
?
0?????????

? ?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643684^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_8643644j;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
,__inference_sequential_layer_call_fn_8643662];?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
0__inference_hidden_layer_0_layer_call_fn_8643691Q0?-
&?#
!?
inputs??????????
? "????????????
-__inference_input_layer_layer_call_fn_8643673P3?0
)?&
$?!
inputs?????????
? "????????????
"__inference__wrapped_model_8643451?>?;
4?1
/?,
input_layer_input?????????
? ";?8
6
output_layer&?#
output_layer?????????
?
,__inference_sequential_layer_call_fn_8643568hF?C
<?9
/?,
input_layer_input?????????
p

 
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643546uF?C
<?9
/?,
input_layer_input?????????
p 

 
? "%?"
?
0?????????

? ?
,__inference_sequential_layer_call_fn_8643653];?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
%__inference_signature_wrapper_8643602?S?P
? 
I?F
D
input_layer_input/?,
input_layer_input?????????";?8
6
output_layer&?#
output_layer?????????
?
,__inference_sequential_layer_call_fn_8643591hF?C
<?9
/?,
input_layer_input?????????
p 

 
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_8643624j;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? 