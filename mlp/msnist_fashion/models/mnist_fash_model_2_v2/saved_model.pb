ǽ
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
shapeshape?"serve*2.0.02unknown8φ
?
hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*&
shared_namehidden_layer_0/kernel*
shape:
??
?
)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel* 
_output_shapes
:
??*
dtype0

hidden_layer_0/biasVarHandleOp*
shape:?*$
shared_namehidden_layer_0/bias*
_output_shapes
: *
dtype0
x
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
dtype0*
_output_shapes	
:?
?
hidden_layer_1/kernelVarHandleOp*&
shared_namehidden_layer_1/kernel*
_output_shapes
: *
shape:
??*
dtype0
?
)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel* 
_output_shapes
:
??*
dtype0

hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namehidden_layer_1/bias
x
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
dtype0*
_output_shapes	
:?
?
output_layer/kernelVarHandleOp*
shape:	?
*$
shared_nameoutput_layer/kernel*
dtype0*
_output_shapes
: 
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	?
*
dtype0
z
output_layer/biasVarHandleOp*
shape:
*
_output_shapes
: *
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
: *
shared_nametotal*
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
shape: *
shared_namecount*
_output_shapes
: *
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
shape:
??*(
shared_namehidden_layer_0/kernel/m*
dtype0
?
+hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/m*
dtype0* 
_output_shapes
:
??
?
hidden_layer_0/bias/mVarHandleOp*
dtype0*
shape:?*&
shared_namehidden_layer_0/bias/m*
_output_shapes
: 
|
)hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/m*
_output_shapes	
:?*
dtype0
?
hidden_layer_1/kernel/mVarHandleOp*
shape:
??*
_output_shapes
: *(
shared_namehidden_layer_1/kernel/m*
dtype0
?
+hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
hidden_layer_1/bias/mVarHandleOp*
shape:?*
dtype0*&
shared_namehidden_layer_1/bias/m*
_output_shapes
: 
|
)hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_1/bias/m*
_output_shapes	
:?*
dtype0
?
output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*&
shared_nameoutput_layer/kernel/m*
shape:	?

?
)output_layer/kernel/m/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/m*
_output_shapes
:	?
*
dtype0
~
output_layer/bias/mVarHandleOp*
_output_shapes
: *
shape:
*$
shared_nameoutput_layer/bias/m*
dtype0
w
'output_layer/bias/m/Read/ReadVariableOpReadVariableOpoutput_layer/bias/m*
_output_shapes
:
*
dtype0
?
hidden_layer_0/kernel/vVarHandleOp*
shape:
??*
_output_shapes
: *
dtype0*(
shared_namehidden_layer_0/kernel/v
?
+hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/v* 
_output_shapes
:
??*
dtype0
?
hidden_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*&
shared_namehidden_layer_0/bias/v*
shape:?
|
)hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/v*
_output_shapes	
:?*
dtype0
?
hidden_layer_1/kernel/vVarHandleOp*
dtype0*(
shared_namehidden_layer_1/kernel/v*
_output_shapes
: *
shape:
??
?
+hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
hidden_layer_1/bias/vVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *&
shared_namehidden_layer_1/bias/v
|
)hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_1/bias/v*
_output_shapes	
:?*
dtype0
?
output_layer/kernel/vVarHandleOp*
shape:	?
*&
shared_nameoutput_layer/kernel/v*
dtype0*
_output_shapes
: 
?
)output_layer/kernel/v/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/v*
dtype0*
_output_shapes
:	?

~
output_layer/bias/vVarHandleOp*
_output_shapes
: *$
shared_nameoutput_layer/bias/v*
shape:
*
dtype0
w
'output_layer/bias/v/Read/ReadVariableOpReadVariableOpoutput_layer/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
?'
ConstConst"/device:CPU:0*
dtype0*?&
value?&B?& B?&
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
lmSmTmUmV%mW&mXvYvZv[v\%v]&v^
*
0
1
2
3
%4
&5
 
*
0
1
2
3
%4
&5
?
	variables
	regularization_losses

+layers

trainable_variables
,non_trainable_variables
-layer_regularization_losses
.metrics
 
 
 
 
?
	variables
regularization_losses

/layers
trainable_variables
0metrics
1layer_regularization_losses
2non_trainable_variables
 
 
 
?
	variables
regularization_losses

3layers
trainable_variables
4metrics
5layer_regularization_losses
6non_trainable_variables
a_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses

7layers
trainable_variables
8metrics
9layer_regularization_losses
:non_trainable_variables
a_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses

;layers
trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
 
 
 
?
!	variables
"regularization_losses

?layers
#trainable_variables
@metrics
Alayer_regularization_losses
Bnon_trainable_variables
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
'	variables
(regularization_losses

Clayers
)trainable_variables
Dmetrics
Elayer_regularization_losses
Fnon_trainable_variables
#
0
1
2
3
4
 
 

G0
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
 
 
 
 
 
 
 
 
x
	Htotal
	Icount
J
_fn_kwargs
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1
 
 
?
K	variables
Lregularization_losses

Olayers
Mtrainable_variables
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
 
 
 

H0
I1
}
VARIABLE_VALUEhidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: 
?
!serving_default_input_layer_inputPlaceholder*
dtype0*+
_output_shapes
:?????????* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputhidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/bias*'
_output_shapes
:?????????
*
Tout
2*
Tin
	2*.
_gradient_op_typePartitionedCall-8644366*.
f)R'
%__inference_signature_wrapper_8644127**
config_proto

GPU 

CPU2J 8
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+hidden_layer_0/kernel/m/Read/ReadVariableOp)hidden_layer_0/bias/m/Read/ReadVariableOp+hidden_layer_1/kernel/m/Read/ReadVariableOp)hidden_layer_1/bias/m/Read/ReadVariableOp)output_layer/kernel/m/Read/ReadVariableOp'output_layer/bias/m/Read/ReadVariableOp+hidden_layer_0/kernel/v/Read/ReadVariableOp)hidden_layer_0/bias/v/Read/ReadVariableOp+hidden_layer_1/kernel/v/Read/ReadVariableOp)hidden_layer_1/bias/v/Read/ReadVariableOp)output_layer/kernel/v/Read/ReadVariableOp'output_layer/bias/v/Read/ReadVariableOpConst*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_8644407*
Tout
2*!
Tin
2*.
_gradient_op_typePartitionedCall-8644408
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biasoutput_layer/kerneloutput_layer/biastotalcounthidden_layer_0/kernel/mhidden_layer_0/bias/mhidden_layer_1/kernel/mhidden_layer_1/bias/moutput_layer/kernel/moutput_layer/bias/mhidden_layer_0/kernel/vhidden_layer_0/bias/vhidden_layer_1/kernel/vhidden_layer_1/bias/voutput_layer/kernel/voutput_layer/bias/v* 
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8644481*
_output_shapes
: *,
f'R%
#__inference__traced_restore_8644480??
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"????  *
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
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_8643994

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:??????????*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
0__inference_hidden_layer_1_layer_call_fn_8644269

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950*
Tout
2*(
_output_shapes
:??????????*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8643956?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?(
?
"__inference__wrapped_model_8643886
input_layer_input<
8sequential_hidden_layer_0_matmul_readvariableop_resource=
9sequential_hidden_layer_0_biasadd_readvariableop_resource<
8sequential_hidden_layer_1_matmul_readvariableop_resource=
9sequential_hidden_layer_1_biasadd_readvariableop_resource:
6sequential_output_layer_matmul_readvariableop_resource;
7sequential_output_layer_biasadd_readvariableop_resource
identity??0sequential/hidden_layer_0/BiasAdd/ReadVariableOp?/sequential/hidden_layer_0/MatMul/ReadVariableOp?0sequential/hidden_layer_1/BiasAdd/ReadVariableOp?/sequential/hidden_layer_1/MatMul/ReadVariableOp?.sequential/output_layer/BiasAdd/ReadVariableOp?-sequential/output_layer/MatMul/ReadVariableOpu
$sequential/input_layer/Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:?
sequential/input_layer/ReshapeReshapeinput_layer_input-sequential/input_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
/sequential/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
 sequential/hidden_layer_0/MatMulMatMul'sequential/input_layer/Reshape:output:07sequential/hidden_layer_0/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
0sequential/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
!sequential/hidden_layer_0/BiasAddBiasAdd*sequential/hidden_layer_0/MatMul:product:08sequential/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
sequential/hidden_layer_0/ReluRelu*sequential/hidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/sequential/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
 sequential/hidden_layer_1/MatMulMatMul,sequential/hidden_layer_0/Relu:activations:07sequential/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
!sequential/hidden_layer_1/BiasAddBiasAdd*sequential/hidden_layer_1/MatMul:product:08sequential/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
sequential/hidden_layer_1/ReluRelu*sequential/hidden_layer_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
sequential/dropout/IdentityIdentity,sequential/hidden_layer_1/Relu:activations:0*(
_output_shapes
:??????????*
T0?
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
sequential/output_layer/MatMulMatMul$sequential/dropout/Identity:output:05sequential/output_layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0?
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential/output_layer/SoftmaxSoftmax(sequential/output_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentity)sequential/output_layer/Softmax:softmax:01^sequential/hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/hidden_layer_0/MatMul/ReadVariableOp1^sequential/hidden_layer_1/BiasAdd/ReadVariableOp0^sequential/hidden_layer_1/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2d
0sequential/hidden_layer_1/BiasAdd/ReadVariableOp0sequential/hidden_layer_1/BiasAdd/ReadVariableOp2d
0sequential/hidden_layer_0/BiasAdd/ReadVariableOp0sequential/hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_1/MatMul/ReadVariableOp/sequential/hidden_layer_1/MatMul/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_0/MatMul/ReadVariableOp/sequential/hidden_layer_0/MatMul/ReadVariableOp: : : : : :1 -
+
_user_specified_nameinput_layer_input: 
?	
?
,__inference_sequential_layer_call_fn_8644114
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8644104*.
_gradient_op_typePartitionedCall-8644105*'
_output_shapes
:?????????
*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: 
?"
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644200

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0r
dropout/IdentityIdentity!hidden_layer_1/Relu:activations:0*(
_output_shapes
:??????????*
T0?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
output_layer/MatMulMatMuldropout/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
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
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?2
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644172

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
valueB"????  *
_output_shapes
:*
dtype0}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0Y
dropout/dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0f
dropout/dropout/ShapeShape!hidden_layer_1/Relu:activations:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*(
_output_shapes
:??????????*
T0?
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:???????????
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*(
_output_shapes
:??????????*
T0?
dropout/dropout/mulMul!hidden_layer_1/Relu:activations:0dropout/dropout/truediv:z:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*(
_output_shapes
:??????????*

DstT0?
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*(
_output_shapes
:??????????*
T0?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
output_layer/MatMulMatMuldropout/dropout/mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?	
?
,__inference_sequential_layer_call_fn_8644085
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*
Tin
	2*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-8644076**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8644075?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644057
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898*
Tout
2*.
_gradient_op_typePartitionedCall-8643904*
Tin
2?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-8643928*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922**
config_proto

GPU 

CPU2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-8643956*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950?
dropout/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-8644006*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643994*
Tin
2*
Tout
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022*.
_gradient_op_typePartitionedCall-8644028*
Tout
2*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?
b
)__inference_dropout_layer_call_fn_8644299

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*(
_output_shapes
:??????????*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643987*
Tin
2*.
_gradient_op_typePartitionedCall-8643998*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644075

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??dropout/StatefulPartitionedCall?&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643904*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643928*
Tout
2?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8643956*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950*
Tin
2?
dropout/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*(
_output_shapes
:??????????*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8643998*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643987?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*'
_output_shapes
:?????????
*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8644028*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022*
Tout
2?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?1
?	
 __inference__traced_save_8644407
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_hidden_layer_0_kernel_m_read_readvariableop4
0savev2_hidden_layer_0_bias_m_read_readvariableop6
2savev2_hidden_layer_1_kernel_m_read_readvariableop4
0savev2_hidden_layer_1_bias_m_read_readvariableop4
0savev2_output_layer_kernel_m_read_readvariableop2
.savev2_output_layer_bias_m_read_readvariableop6
2savev2_hidden_layer_0_kernel_v_read_readvariableop4
0savev2_hidden_layer_0_bias_v_read_readvariableop6
2savev2_hidden_layer_1_kernel_v_read_readvariableop4
0savev2_hidden_layer_1_bias_v_read_readvariableop4
0savev2_output_layer_kernel_v_read_readvariableop2
.savev2_output_layer_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_79a86d17f35144c793c7a133f7e9140f/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_hidden_layer_0_kernel_m_read_readvariableop0savev2_hidden_layer_0_bias_m_read_readvariableop2savev2_hidden_layer_1_kernel_m_read_readvariableop0savev2_hidden_layer_1_bias_m_read_readvariableop0savev2_output_layer_kernel_m_read_readvariableop.savev2_output_layer_bias_m_read_readvariableop2savev2_hidden_layer_0_kernel_v_read_readvariableop0savev2_hidden_layer_0_bias_v_read_readvariableop2savev2_hidden_layer_1_kernel_v_read_readvariableop0savev2_hidden_layer_1_bias_v_read_readvariableop0savev2_output_layer_kernel_v_read_readvariableop.savev2_output_layer_bias_v_read_readvariableop"/device:CPU:0*"
dtypes
2*
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
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
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
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?
:
: : :
??:?:
??:?:	?
:
:
??:?:
??:?:	?
:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : 
?
?
.__inference_output_layer_layer_call_fn_8644322

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022*
Tout
2*
Tin
2*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-8644028**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8644262

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
,__inference_sequential_layer_call_fn_8644222

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8644104*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-8644105**
config_proto

GPU 

CPU2J 8*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: 
?R
?
#__inference__traced_restore_8644480
file_prefix*
&assignvariableop_hidden_layer_0_kernel*
&assignvariableop_1_hidden_layer_0_bias,
(assignvariableop_2_hidden_layer_1_kernel*
&assignvariableop_3_hidden_layer_1_bias*
&assignvariableop_4_output_layer_kernel(
$assignvariableop_5_output_layer_bias
assignvariableop_6_total
assignvariableop_7_count.
*assignvariableop_8_hidden_layer_0_kernel_m,
(assignvariableop_9_hidden_layer_0_bias_m/
+assignvariableop_10_hidden_layer_1_kernel_m-
)assignvariableop_11_hidden_layer_1_bias_m-
)assignvariableop_12_output_layer_kernel_m+
'assignvariableop_13_output_layer_bias_m/
+assignvariableop_14_hidden_layer_0_kernel_v-
)assignvariableop_15_hidden_layer_0_bias_v/
+assignvariableop_16_hidden_layer_1_kernel_v-
)assignvariableop_17_hidden_layer_1_bias_v-
)assignvariableop_18_output_layer_kernel_v+
'assignvariableop_19_output_layer_bias_v
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_output_layer_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_output_layer_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0x
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0x
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_hidden_layer_0_kernel_mIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_hidden_layer_0_bias_mIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_hidden_layer_1_kernel_mIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_hidden_layer_1_bias_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_output_layer_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_output_layer_bias_mIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_hidden_layer_0_kernel_vIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_hidden_layer_0_bias_vIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_hidden_layer_1_kernel_vIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_hidden_layer_1_bias_vIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_output_layer_kernel_vIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_output_layer_bias_vIdentity_19:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644040
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??dropout/StatefulPartitionedCall?&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*.
_gradient_op_typePartitionedCall-8643904*(
_output_shapes
:??????????*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922*
Tout
2*.
_gradient_op_typePartitionedCall-8643928*(
_output_shapes
:???????????
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*
Tout
2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950*
Tin
2*.
_gradient_op_typePartitionedCall-8643956**
config_proto

GPU 

CPU2J 8?
dropout/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643987*
Tin
2*.
_gradient_op_typePartitionedCall-8643998*
Tout
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tout
2*.
_gradient_op_typePartitionedCall-8644028*
Tin
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?
?
0__inference_hidden_layer_0_layer_call_fn_8644251

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922*
Tin
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tout
2*.
_gradient_op_typePartitionedCall-8643928?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
,__inference_sequential_layer_call_fn_8644211

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*.
_gradient_op_typePartitionedCall-8644076**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_8644075?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: 
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_8644294

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
i
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_8644315

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
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
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
I
-__inference_input_layer_layer_call_fn_8644233

inputs
identity?
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898*
Tout
2*
Tin
2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8643904a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
E
)__inference_dropout_layer_call_fn_8644304

inputs
identity?
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-8644006*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643994*
Tout
2*
Tin
2a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_8644127
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*+
f&R$
"__inference__wrapped_model_8643886**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8644118*'
_output_shapes
:?????????
*
Tin
	2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8644244

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_8643987

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *   ?*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:??????????*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*(
_output_shapes
:??????????*

SrcT0
*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:??????????*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_8644289

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *   ?*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*(
_output_shapes
:??????????*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:???????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:??????????*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:??????????*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:??????????j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644104

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*.
_gradient_op_typePartitionedCall-8643904**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_8643898?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8643922*.
_gradient_op_typePartitionedCall-8643928**
config_proto

GPU 

CPU2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8643950*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-8643956?
dropout/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-8644006*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_8643994*
Tout
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-8644028**
config_proto

GPU 

CPU2J 8*
Tout
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_8644022*
Tin
2*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_8644228

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"????  *
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs"wL
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
tensorflow/serving/predict:ۯ
?"
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
__default_save_signature
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 608, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 608}}}}
?
!	variables
"regularization_losses
#trainable_variables
$	keras_api
*j&call_and_return_all_conditional_losses
k__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
*l&call_and_return_all_conditional_losses
m__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 608}}}}
mSmTmUmV%mW&mXvYvZv[v\%v]&v^"
	optimizer
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
?
	variables
	regularization_losses

+layers

trainable_variables
,non_trainable_variables
-layer_regularization_losses
.metrics
a__call__
__default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
nserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses

/layers
trainable_variables
0metrics
1layer_regularization_losses
2non_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses

3layers
trainable_variables
4metrics
5layer_regularization_losses
6non_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_0/kernel
": ?2hidden_layer_0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses

7layers
trainable_variables
8metrics
9layer_regularization_losses
:non_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_1/kernel
": ?2hidden_layer_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses

;layers
trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!	variables
"regularization_losses

?layers
#trainable_variables
@metrics
Alayer_regularization_losses
Bnon_trainable_variables
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
&:$	?
2output_layer/kernel
:
2output_layer/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'	variables
(regularization_losses

Clayers
)trainable_variables
Dmetrics
Elayer_regularization_losses
Fnon_trainable_variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
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
	Htotal
	Icount
J
_fn_kwargs
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
K	variables
Lregularization_losses

Olayers
Mtrainable_variables
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
):'
??2hidden_layer_0/kernel/m
": ?2hidden_layer_0/bias/m
):'
??2hidden_layer_1/kernel/m
": ?2hidden_layer_1/bias/m
&:$	?
2output_layer/kernel/m
:
2output_layer/bias/m
):'
??2hidden_layer_0/kernel/v
": ?2hidden_layer_0/bias/v
):'
??2hidden_layer_1/kernel/v
": ?2hidden_layer_1/bias/v
&:$	?
2output_layer/kernel/v
:
2output_layer/bias/v
?2?
"__inference__wrapped_model_8643886?
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
G__inference_sequential_layer_call_and_return_conditional_losses_8644057
G__inference_sequential_layer_call_and_return_conditional_losses_8644200
G__inference_sequential_layer_call_and_return_conditional_losses_8644172
G__inference_sequential_layer_call_and_return_conditional_losses_8644040?
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
,__inference_sequential_layer_call_fn_8644222
,__inference_sequential_layer_call_fn_8644085
,__inference_sequential_layer_call_fn_8644211
,__inference_sequential_layer_call_fn_8644114?
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
H__inference_input_layer_layer_call_and_return_conditional_losses_8644228?
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
-__inference_input_layer_layer_call_fn_8644233?
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8644244?
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
0__inference_hidden_layer_0_layer_call_fn_8644251?
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
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8644262?
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
0__inference_hidden_layer_1_layer_call_fn_8644269?
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
?2?
D__inference_dropout_layer_call_and_return_conditional_losses_8644294
D__inference_dropout_layer_call_and_return_conditional_losses_8644289?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_layer_call_fn_8644304
)__inference_dropout_layer_call_fn_8644299?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_output_layer_layer_call_and_return_conditional_losses_8644315?
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
.__inference_output_layer_layer_call_fn_8644322?
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
%__inference_signature_wrapper_8644127input_layer_input
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
,__inference_sequential_layer_call_fn_8644085j%&F?C
<?9
/?,
input_layer_input?????????
p

 
? "??????????
?
,__inference_sequential_layer_call_fn_8644114j%&F?C
<?9
/?,
input_layer_input?????????
p 

 
? "??????????
?
H__inference_input_layer_layer_call_and_return_conditional_losses_8644228]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
"__inference__wrapped_model_8643886?%&>?;
4?1
/?,
input_layer_input?????????
? ";?8
6
output_layer&?#
output_layer?????????
?
D__inference_dropout_layer_call_and_return_conditional_losses_8644294^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_8644244^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_hidden_layer_0_layer_call_fn_8644251Q0?-
&?#
!?
inputs??????????
? "???????????~
)__inference_dropout_layer_call_fn_8644299Q4?1
*?'
!?
inputs??????????
p
? "????????????
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_8644262^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_sequential_layer_call_fn_8644211_%&;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
,__inference_sequential_layer_call_fn_8644222_%&;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
I__inference_output_layer_layer_call_and_return_conditional_losses_8644315]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
D__inference_dropout_layer_call_and_return_conditional_losses_8644289^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
-__inference_input_layer_layer_call_fn_8644233P3?0
)?&
$?!
inputs?????????
? "????????????
G__inference_sequential_layer_call_and_return_conditional_losses_8644057w%&F?C
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
G__inference_sequential_layer_call_and_return_conditional_losses_8644200l%&;?8
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
? ~
)__inference_dropout_layer_call_fn_8644304Q4?1
*?'
!?
inputs??????????
p 
? "????????????
.__inference_output_layer_layer_call_fn_8644322P%&0?-
&?#
!?
inputs??????????
? "??????????
?
0__inference_hidden_layer_1_layer_call_fn_8644269Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_sequential_layer_call_and_return_conditional_losses_8644040w%&F?C
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
%__inference_signature_wrapper_8644127?%&S?P
? 
I?F
D
input_layer_input/?,
input_layer_input?????????";?8
6
output_layer&?#
output_layer?????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_8644172l%&;?8
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