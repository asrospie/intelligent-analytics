íî

Ô
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ªÇ	
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
u
enc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*
shared_nameenc_3/kernel
n
 enc_3/kernel/Read/ReadVariableOpReadVariableOpenc_3/kernel*
_output_shapes
:	Äb*
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
t
dec_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1b*
shared_namedec_4/kernel
m
 dec_4/kernel/Read/ReadVariableOpReadVariableOpdec_4/kernel*
_output_shapes

:1b*
dtype0
l

dec_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*
shared_name
dec_4/bias
e
dec_4/bias/Read/ReadVariableOpReadVariableOp
dec_4/bias*
_output_shapes
:b*
dtype0
u
dec_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*
shared_namedec_3/kernel
n
 dec_3/kernel/Read/ReadVariableOpReadVariableOpdec_3/kernel*
_output_shapes
:	bÄ*
dtype0
m

dec_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*
shared_name
dec_3/bias
f
dec_3/bias/Read/ReadVariableOpReadVariableOp
dec_3/bias*
_output_shapes	
:Ä*
dtype0
v
dec_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*
shared_namedec_2/kernel
o
 dec_2/kernel/Read/ReadVariableOpReadVariableOpdec_2/kernel* 
_output_shapes
:
Ä*
dtype0
m

dec_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dec_2/bias
f
dec_2/bias/Read/ReadVariableOpReadVariableOp
dec_2/bias*
_output_shapes	
:*
dtype0

decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namedecoder_output/kernel

)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel* 
_output_shapes
:
*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

Adam/enc_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/enc_1/kernel/m
}
'Adam/enc_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_1/kernel/m* 
_output_shapes
:
*
dtype0
{
Adam/enc_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/enc_1/bias/m
t
%Adam/enc_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_1/bias/m*
_output_shapes	
:*
dtype0

Adam/enc_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/enc_2/kernel/m
}
'Adam/enc_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_2/kernel/m* 
_output_shapes
:
Ä*
dtype0
{
Adam/enc_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/enc_2/bias/m
t
%Adam/enc_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_2/bias/m*
_output_shapes	
:Ä*
dtype0

Adam/enc_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*$
shared_nameAdam/enc_3/kernel/m
|
'Adam/enc_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_3/kernel/m*
_output_shapes
:	Äb*
dtype0
z
Adam/enc_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/enc_3/bias/m
s
%Adam/enc_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_3/bias/m*
_output_shapes
:b*
dtype0

Adam/enc_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:b1*$
shared_nameAdam/enc_4/kernel/m
{
'Adam/enc_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_4/kernel/m*
_output_shapes

:b1*
dtype0
z
Adam/enc_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_nameAdam/enc_4/bias/m
s
%Adam/enc_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_4/bias/m*
_output_shapes
:1*
dtype0

Adam/dec_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1b*$
shared_nameAdam/dec_4/kernel/m
{
'Adam/dec_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_4/kernel/m*
_output_shapes

:1b*
dtype0
z
Adam/dec_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/dec_4/bias/m
s
%Adam/dec_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_4/bias/m*
_output_shapes
:b*
dtype0

Adam/dec_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*$
shared_nameAdam/dec_3/kernel/m
|
'Adam/dec_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_3/kernel/m*
_output_shapes
:	bÄ*
dtype0
{
Adam/dec_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/dec_3/bias/m
t
%Adam/dec_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_3/bias/m*
_output_shapes	
:Ä*
dtype0

Adam/dec_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/dec_2/kernel/m
}
'Adam/dec_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_2/kernel/m* 
_output_shapes
:
Ä*
dtype0
{
Adam/dec_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dec_2/bias/m
t
%Adam/dec_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_2/bias/m*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/m

0Adam/decoder_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/m* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/m

.Adam/decoder_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/m*
_output_shapes	
:*
dtype0

Adam/enc_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/enc_1/kernel/v
}
'Adam/enc_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_1/kernel/v* 
_output_shapes
:
*
dtype0
{
Adam/enc_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/enc_1/bias/v
t
%Adam/enc_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_1/bias/v*
_output_shapes	
:*
dtype0

Adam/enc_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/enc_2/kernel/v
}
'Adam/enc_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_2/kernel/v* 
_output_shapes
:
Ä*
dtype0
{
Adam/enc_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/enc_2/bias/v
t
%Adam/enc_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_2/bias/v*
_output_shapes	
:Ä*
dtype0

Adam/enc_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*$
shared_nameAdam/enc_3/kernel/v
|
'Adam/enc_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_3/kernel/v*
_output_shapes
:	Äb*
dtype0
z
Adam/enc_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/enc_3/bias/v
s
%Adam/enc_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_3/bias/v*
_output_shapes
:b*
dtype0

Adam/enc_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:b1*$
shared_nameAdam/enc_4/kernel/v
{
'Adam/enc_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_4/kernel/v*
_output_shapes

:b1*
dtype0
z
Adam/enc_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_nameAdam/enc_4/bias/v
s
%Adam/enc_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_4/bias/v*
_output_shapes
:1*
dtype0

Adam/dec_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1b*$
shared_nameAdam/dec_4/kernel/v
{
'Adam/dec_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_4/kernel/v*
_output_shapes

:1b*
dtype0
z
Adam/dec_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/dec_4/bias/v
s
%Adam/dec_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_4/bias/v*
_output_shapes
:b*
dtype0

Adam/dec_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*$
shared_nameAdam/dec_3/kernel/v
|
'Adam/dec_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_3/kernel/v*
_output_shapes
:	bÄ*
dtype0
{
Adam/dec_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/dec_3/bias/v
t
%Adam/dec_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_3/bias/v*
_output_shapes	
:Ä*
dtype0

Adam/dec_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/dec_2/kernel/v
}
'Adam/dec_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_2/kernel/v* 
_output_shapes
:
Ä*
dtype0
{
Adam/dec_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dec_2/bias/v
t
%Adam/dec_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_2/bias/v*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/v

0Adam/decoder_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/v* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/v

.Adam/decoder_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ÿ^
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*º^
value°^B­^ B¦^

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
¦

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*
¦

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
¦

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
¦

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*
¦

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses*

Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemmmm#m$m+m,m3m4m;m<mCmDmKmLmvvvv#v$v +v¡,v¢3v£4v¤;v¥<v¦Cv§Dv¨Kv©Lvª*
z
0
1
2
3
#4
$5
+6
,7
38
49
;10
<11
C12
D13
K14
L15*
z
0
1
2
3
#4
$5
+6
,7
38
49
;10
<11
C12
D13
K14
L15*
* 
°
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

]serving_default* 
\V
VARIABLE_VALUEenc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
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

#0
$1*

#0
$1*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdec_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dec_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdec_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dec_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdec_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dec_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

K0
L1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

0*
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
<

total

count
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
y
VARIABLE_VALUEAdam/enc_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/decoder_output/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/decoder_output/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/decoder_output/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/decoder_output/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_layerPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¹
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasenc_4/kernel
enc_4/biasdec_4/kernel
dec_4/biasdec_3/kernel
dec_3/biasdec_2/kernel
dec_2/biasdecoder_output/kerneldecoder_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_147983
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp enc_3/kernel/Read/ReadVariableOpenc_3/bias/Read/ReadVariableOp enc_4/kernel/Read/ReadVariableOpenc_4/bias/Read/ReadVariableOp dec_4/kernel/Read/ReadVariableOpdec_4/bias/Read/ReadVariableOp dec_3/kernel/Read/ReadVariableOpdec_3/bias/Read/ReadVariableOp dec_2/kernel/Read/ReadVariableOpdec_2/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/enc_1/kernel/m/Read/ReadVariableOp%Adam/enc_1/bias/m/Read/ReadVariableOp'Adam/enc_2/kernel/m/Read/ReadVariableOp%Adam/enc_2/bias/m/Read/ReadVariableOp'Adam/enc_3/kernel/m/Read/ReadVariableOp%Adam/enc_3/bias/m/Read/ReadVariableOp'Adam/enc_4/kernel/m/Read/ReadVariableOp%Adam/enc_4/bias/m/Read/ReadVariableOp'Adam/dec_4/kernel/m/Read/ReadVariableOp%Adam/dec_4/bias/m/Read/ReadVariableOp'Adam/dec_3/kernel/m/Read/ReadVariableOp%Adam/dec_3/bias/m/Read/ReadVariableOp'Adam/dec_2/kernel/m/Read/ReadVariableOp%Adam/dec_2/bias/m/Read/ReadVariableOp0Adam/decoder_output/kernel/m/Read/ReadVariableOp.Adam/decoder_output/bias/m/Read/ReadVariableOp'Adam/enc_1/kernel/v/Read/ReadVariableOp%Adam/enc_1/bias/v/Read/ReadVariableOp'Adam/enc_2/kernel/v/Read/ReadVariableOp%Adam/enc_2/bias/v/Read/ReadVariableOp'Adam/enc_3/kernel/v/Read/ReadVariableOp%Adam/enc_3/bias/v/Read/ReadVariableOp'Adam/enc_4/kernel/v/Read/ReadVariableOp%Adam/enc_4/bias/v/Read/ReadVariableOp'Adam/dec_4/kernel/v/Read/ReadVariableOp%Adam/dec_4/bias/v/Read/ReadVariableOp'Adam/dec_3/kernel/v/Read/ReadVariableOp%Adam/dec_3/bias/v/Read/ReadVariableOp'Adam/dec_2/kernel/v/Read/ReadVariableOp%Adam/dec_2/bias/v/Read/ReadVariableOp0Adam/decoder_output/kernel/v/Read/ReadVariableOp.Adam/decoder_output/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_148331
¦

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasenc_4/kernel
enc_4/biasdec_4/kernel
dec_4/biasdec_3/kernel
dec_3/biasdec_2/kernel
dec_2/biasdecoder_output/kerneldecoder_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/enc_1/kernel/mAdam/enc_1/bias/mAdam/enc_2/kernel/mAdam/enc_2/bias/mAdam/enc_3/kernel/mAdam/enc_3/bias/mAdam/enc_4/kernel/mAdam/enc_4/bias/mAdam/dec_4/kernel/mAdam/dec_4/bias/mAdam/dec_3/kernel/mAdam/dec_3/bias/mAdam/dec_2/kernel/mAdam/dec_2/bias/mAdam/decoder_output/kernel/mAdam/decoder_output/bias/mAdam/enc_1/kernel/vAdam/enc_1/bias/vAdam/enc_2/kernel/vAdam/enc_2/bias/vAdam/enc_3/kernel/vAdam/enc_3/bias/vAdam/enc_4/kernel/vAdam/enc_4/bias/vAdam/dec_4/kernel/vAdam/dec_4/bias/vAdam/dec_3/kernel/vAdam/dec_3/bias/vAdam/dec_2/kernel/vAdam/dec_2/bias/vAdam/decoder_output/kernel/vAdam/decoder_output/bias/v*C
Tin<
:28*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_148506Ý


ó
A__inference_enc_3_layer_call_and_return_conditional_losses_147294

inputs1
matmul_readvariableop_resource:	Äb-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
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


ò
A__inference_dec_4_layer_call_and_return_conditional_losses_147328

inputs0
matmul_readvariableop_resource:1b-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1b*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
C
å
C__inference_model_3_layer_call_and_return_conditional_losses_147944

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	Ä7
$enc_3_matmul_readvariableop_resource:	Äb3
%enc_3_biasadd_readvariableop_resource:b6
$enc_4_matmul_readvariableop_resource:b13
%enc_4_biasadd_readvariableop_resource:16
$dec_4_matmul_readvariableop_resource:1b3
%dec_4_biasadd_readvariableop_resource:b7
$dec_3_matmul_readvariableop_resource:	bÄ4
%dec_3_biasadd_readvariableop_resource:	Ä8
$dec_2_matmul_readvariableop_resource:
Ä4
%dec_2_biasadd_readvariableop_resource:	A
-decoder_output_matmul_readvariableop_resource:
=
.decoder_output_biasadd_readvariableop_resource:	
identity¢dec_2/BiasAdd/ReadVariableOp¢dec_2/MatMul/ReadVariableOp¢dec_3/BiasAdd/ReadVariableOp¢dec_3/MatMul/ReadVariableOp¢dec_4/BiasAdd/ReadVariableOp¢dec_4/MatMul/ReadVariableOp¢%decoder_output/BiasAdd/ReadVariableOp¢$decoder_output/MatMul/ReadVariableOp¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢enc_3/BiasAdd/ReadVariableOp¢enc_3/MatMul/ReadVariableOp¢enc_4/BiasAdd/ReadVariableOp¢enc_4/MatMul/ReadVariableOp
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
:ÿÿÿÿÿÿÿÿÿÄ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
enc_4/MatMul/ReadVariableOpReadVariableOp$enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0
enc_4/MatMulMatMulenc_3/Relu:activations:0#enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
enc_4/BiasAdd/ReadVariableOpReadVariableOp%enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
enc_4/BiasAddBiasAddenc_4/MatMul:product:0$enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\

enc_4/ReluReluenc_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dec_4/MatMul/ReadVariableOpReadVariableOp$dec_4_matmul_readvariableop_resource*
_output_shapes

:1b*
dtype0
dec_4/MatMulMatMulenc_4/Relu:activations:0#dec_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
dec_4/BiasAdd/ReadVariableOpReadVariableOp%dec_4_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
dec_4/BiasAddBiasAdddec_4/MatMul:product:0$dec_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

dec_4/ReluReludec_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dec_3/MatMul/ReadVariableOpReadVariableOp$dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0
dec_3/MatMulMatMuldec_4/Relu:activations:0#dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_3/BiasAdd/ReadVariableOpReadVariableOp%dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
dec_3/BiasAddBiasAdddec_3/MatMul:product:0$dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

dec_3/ReluReludec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_2/MatMul/ReadVariableOpReadVariableOp$dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
dec_2/MatMulMatMuldec_3/Relu:activations:0#dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dec_2/BiasAdd/ReadVariableOpReadVariableOp%dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dec_2/BiasAddBiasAdddec_2/MatMul:product:0$dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dec_2/ReluReludec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
decoder_output/MatMulMatMuldec_2/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dec_2/BiasAdd/ReadVariableOp^dec_2/MatMul/ReadVariableOp^dec_3/BiasAdd/ReadVariableOp^dec_3/MatMul/ReadVariableOp^dec_4/BiasAdd/ReadVariableOp^dec_4/MatMul/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp^enc_4/BiasAdd/ReadVariableOp^enc_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2<
dec_2/BiasAdd/ReadVariableOpdec_2/BiasAdd/ReadVariableOp2:
dec_2/MatMul/ReadVariableOpdec_2/MatMul/ReadVariableOp2<
dec_3/BiasAdd/ReadVariableOpdec_3/BiasAdd/ReadVariableOp2:
dec_3/MatMul/ReadVariableOpdec_3/MatMul/ReadVariableOp2<
dec_4/BiasAdd/ReadVariableOpdec_4/BiasAdd/ReadVariableOp2:
dec_4/MatMul/ReadVariableOpdec_4/MatMul/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2<
enc_4/BiasAdd/ReadVariableOpenc_4/BiasAdd/ReadVariableOp2:
enc_4/MatMul/ReadVariableOpenc_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿

&__inference_enc_3_layer_call_fn_148032

inputs
unknown:	Äb
	unknown_0:b
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_147294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb`
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
 

ô
A__inference_dec_3_layer_call_and_return_conditional_losses_147345

inputs1
matmul_readvariableop_resource:	bÄ.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	bÄ*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¤

õ
A__inference_enc_2_layer_call_and_return_conditional_losses_148023

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
¤

õ
A__inference_enc_1_layer_call_and_return_conditional_losses_148003

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
Õ

/__inference_decoder_output_layer_call_fn_148132

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
ù(
ì
C__inference_model_3_layer_call_and_return_conditional_losses_147584

inputs 
enc_1_147543:

enc_1_147545:	 
enc_2_147548:
Ä
enc_2_147550:	Ä
enc_3_147553:	Äb
enc_3_147555:b
enc_4_147558:b1
enc_4_147560:1
dec_4_147563:1b
dec_4_147565:b
dec_3_147568:	bÄ
dec_3_147570:	Ä 
dec_2_147573:
Ä
dec_2_147575:	)
decoder_output_147578:
$
decoder_output_147580:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢dec_4/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCall¢enc_4/StatefulPartitionedCallå
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_147543enc_1_147545*
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
GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_147260
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_147548enc_2_147550*
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
GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_147277
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_147553enc_3_147555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_147294
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_147558enc_4_147560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_4_layer_call_and_return_conditional_losses_147311
dec_4/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0dec_4_147563dec_4_147565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_4_layer_call_and_return_conditional_losses_147328
dec_3/StatefulPartitionedCallStatefulPartitionedCall&dec_4/StatefulPartitionedCall:output:0dec_3_147568dec_3_147570*
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
GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_147345
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_147573dec_2_147575*
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
GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_147362©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_147578decoder_output_147580*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall^dec_4/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2>
dec_4/StatefulPartitionedCalldec_4/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
£
(__inference_model_3_layer_call_fn_147787

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1b
	unknown_8:b
	unknown_9:	bÄ

unknown_10:	Ä

unknown_11:
Ä

unknown_12:	

unknown_13:


unknown_14:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_147386p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
¨
(__inference_model_3_layer_call_fn_147656
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1b
	unknown_8:b
	unknown_9:	bÄ

unknown_10:	Ä

unknown_11:
Ä

unknown_12:	

unknown_13:


unknown_14:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_147584p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
ù(
ì
C__inference_model_3_layer_call_and_return_conditional_losses_147386

inputs 
enc_1_147261:

enc_1_147263:	 
enc_2_147278:
Ä
enc_2_147280:	Ä
enc_3_147295:	Äb
enc_3_147297:b
enc_4_147312:b1
enc_4_147314:1
dec_4_147329:1b
dec_4_147331:b
dec_3_147346:	bÄ
dec_3_147348:	Ä 
dec_2_147363:
Ä
dec_2_147365:	)
decoder_output_147380:
$
decoder_output_147382:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢dec_4/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCall¢enc_4/StatefulPartitionedCallå
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_147261enc_1_147263*
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
GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_147260
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_147278enc_2_147280*
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
GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_147277
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_147295enc_3_147297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_147294
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_147312enc_4_147314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_4_layer_call_and_return_conditional_losses_147311
dec_4/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0dec_4_147329dec_4_147331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_4_layer_call_and_return_conditional_losses_147328
dec_3/StatefulPartitionedCallStatefulPartitionedCall&dec_4/StatefulPartitionedCall:output:0dec_3_147346dec_3_147348*
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
GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_147345
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_147363dec_2_147365*
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
GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_147362©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_147380decoder_output_147382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall^dec_4/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2>
dec_4/StatefulPartitionedCalldec_4/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
C
å
C__inference_model_3_layer_call_and_return_conditional_losses_147884

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	Ä7
$enc_3_matmul_readvariableop_resource:	Äb3
%enc_3_biasadd_readvariableop_resource:b6
$enc_4_matmul_readvariableop_resource:b13
%enc_4_biasadd_readvariableop_resource:16
$dec_4_matmul_readvariableop_resource:1b3
%dec_4_biasadd_readvariableop_resource:b7
$dec_3_matmul_readvariableop_resource:	bÄ4
%dec_3_biasadd_readvariableop_resource:	Ä8
$dec_2_matmul_readvariableop_resource:
Ä4
%dec_2_biasadd_readvariableop_resource:	A
-decoder_output_matmul_readvariableop_resource:
=
.decoder_output_biasadd_readvariableop_resource:	
identity¢dec_2/BiasAdd/ReadVariableOp¢dec_2/MatMul/ReadVariableOp¢dec_3/BiasAdd/ReadVariableOp¢dec_3/MatMul/ReadVariableOp¢dec_4/BiasAdd/ReadVariableOp¢dec_4/MatMul/ReadVariableOp¢%decoder_output/BiasAdd/ReadVariableOp¢$decoder_output/MatMul/ReadVariableOp¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢enc_3/BiasAdd/ReadVariableOp¢enc_3/MatMul/ReadVariableOp¢enc_4/BiasAdd/ReadVariableOp¢enc_4/MatMul/ReadVariableOp
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
:ÿÿÿÿÿÿÿÿÿÄ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
enc_4/MatMul/ReadVariableOpReadVariableOp$enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0
enc_4/MatMulMatMulenc_3/Relu:activations:0#enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
enc_4/BiasAdd/ReadVariableOpReadVariableOp%enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0
enc_4/BiasAddBiasAddenc_4/MatMul:product:0$enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\

enc_4/ReluReluenc_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dec_4/MatMul/ReadVariableOpReadVariableOp$dec_4_matmul_readvariableop_resource*
_output_shapes

:1b*
dtype0
dec_4/MatMulMatMulenc_4/Relu:activations:0#dec_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
dec_4/BiasAdd/ReadVariableOpReadVariableOp%dec_4_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
dec_4/BiasAddBiasAdddec_4/MatMul:product:0$dec_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

dec_4/ReluReludec_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dec_3/MatMul/ReadVariableOpReadVariableOp$dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0
dec_3/MatMulMatMuldec_4/Relu:activations:0#dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_3/BiasAdd/ReadVariableOpReadVariableOp%dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
dec_3/BiasAddBiasAdddec_3/MatMul:product:0$dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

dec_3/ReluReludec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_2/MatMul/ReadVariableOpReadVariableOp$dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
dec_2/MatMulMatMuldec_3/Relu:activations:0#dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dec_2/BiasAdd/ReadVariableOpReadVariableOp%dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dec_2/BiasAddBiasAdddec_2/MatMul:product:0$dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dec_2/ReluReludec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
decoder_output/MatMulMatMuldec_2/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dec_2/BiasAdd/ReadVariableOp^dec_2/MatMul/ReadVariableOp^dec_3/BiasAdd/ReadVariableOp^dec_3/MatMul/ReadVariableOp^dec_4/BiasAdd/ReadVariableOp^dec_4/MatMul/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp^enc_4/BiasAdd/ReadVariableOp^enc_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2<
dec_2/BiasAdd/ReadVariableOpdec_2/BiasAdd/ReadVariableOp2:
dec_2/MatMul/ReadVariableOpdec_2/MatMul/ReadVariableOp2<
dec_3/BiasAdd/ReadVariableOpdec_3/BiasAdd/ReadVariableOp2:
dec_3/MatMul/ReadVariableOpdec_3/MatMul/ReadVariableOp2<
dec_4/BiasAdd/ReadVariableOpdec_4/BiasAdd/ReadVariableOp2:
dec_4/MatMul/ReadVariableOpdec_4/MatMul/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp2<
enc_4/BiasAdd/ReadVariableOpenc_4/BiasAdd/ReadVariableOp2:
enc_4/MatMul/ReadVariableOpenc_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

&__inference_dec_2_layer_call_fn_148112

inputs
unknown:
Ä
	unknown_0:	
identity¢StatefulPartitionedCall×
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
GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_147362p
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
:ÿÿÿÿÿÿÿÿÿÄ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
©
£
(__inference_model_3_layer_call_fn_147824

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1b
	unknown_8:b
	unknown_9:	bÄ

unknown_10:	Ä

unknown_11:
Ä

unknown_12:	

unknown_13:


unknown_14:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_147584p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ò
A__inference_dec_4_layer_call_and_return_conditional_losses_148083

inputs0
matmul_readvariableop_resource:1b-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1b*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¬

þ
J__inference_decoder_output_layer_call_and_return_conditional_losses_148143

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
¤

õ
A__inference_dec_2_layer_call_and_return_conditional_losses_147362

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
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
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
Ã

&__inference_enc_2_layer_call_fn_148012

inputs
unknown:
Ä
	unknown_0:	Ä
identity¢StatefulPartitionedCall×
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
GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_147277p
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
Ã

&__inference_enc_1_layer_call_fn_147992

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCall×
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
GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_147260p
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
¤

õ
A__inference_enc_1_layer_call_and_return_conditional_losses_147260

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


ò
A__inference_enc_4_layer_call_and_return_conditional_losses_147311

inputs0
matmul_readvariableop_resource:b1-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:b1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¤

õ
A__inference_dec_2_layer_call_and_return_conditional_losses_148123

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
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
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¬

þ
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
 

ô
A__inference_dec_3_layer_call_and_return_conditional_losses_148103

inputs1
matmul_readvariableop_resource:	bÄ.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	bÄ*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¼

&__inference_dec_4_layer_call_fn_148072

inputs
unknown:1b
	unknown_0:b
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_4_layer_call_and_return_conditional_losses_147328o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ÿk
î
__inference__traced_save_148331
file_prefix+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop+
'savev2_enc_3_kernel_read_readvariableop)
%savev2_enc_3_bias_read_readvariableop+
'savev2_enc_4_kernel_read_readvariableop)
%savev2_enc_4_bias_read_readvariableop+
'savev2_dec_4_kernel_read_readvariableop)
%savev2_dec_4_bias_read_readvariableop+
'savev2_dec_3_kernel_read_readvariableop)
%savev2_dec_3_bias_read_readvariableop+
'savev2_dec_2_kernel_read_readvariableop)
%savev2_dec_2_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_enc_1_kernel_m_read_readvariableop0
,savev2_adam_enc_1_bias_m_read_readvariableop2
.savev2_adam_enc_2_kernel_m_read_readvariableop0
,savev2_adam_enc_2_bias_m_read_readvariableop2
.savev2_adam_enc_3_kernel_m_read_readvariableop0
,savev2_adam_enc_3_bias_m_read_readvariableop2
.savev2_adam_enc_4_kernel_m_read_readvariableop0
,savev2_adam_enc_4_bias_m_read_readvariableop2
.savev2_adam_dec_4_kernel_m_read_readvariableop0
,savev2_adam_dec_4_bias_m_read_readvariableop2
.savev2_adam_dec_3_kernel_m_read_readvariableop0
,savev2_adam_dec_3_bias_m_read_readvariableop2
.savev2_adam_dec_2_kernel_m_read_readvariableop0
,savev2_adam_dec_2_bias_m_read_readvariableop;
7savev2_adam_decoder_output_kernel_m_read_readvariableop9
5savev2_adam_decoder_output_bias_m_read_readvariableop2
.savev2_adam_enc_1_kernel_v_read_readvariableop0
,savev2_adam_enc_1_bias_v_read_readvariableop2
.savev2_adam_enc_2_kernel_v_read_readvariableop0
,savev2_adam_enc_2_bias_v_read_readvariableop2
.savev2_adam_enc_3_kernel_v_read_readvariableop0
,savev2_adam_enc_3_bias_v_read_readvariableop2
.savev2_adam_enc_4_kernel_v_read_readvariableop0
,savev2_adam_enc_4_bias_v_read_readvariableop2
.savev2_adam_dec_4_kernel_v_read_readvariableop0
,savev2_adam_dec_4_bias_v_read_readvariableop2
.savev2_adam_dec_3_kernel_v_read_readvariableop0
,savev2_adam_dec_3_bias_v_read_readvariableop2
.savev2_adam_dec_2_kernel_v_read_readvariableop0
,savev2_adam_dec_2_bias_v_read_readvariableop;
7savev2_adam_decoder_output_kernel_v_read_readvariableop9
5savev2_adam_decoder_output_bias_v_read_readvariableop
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
: «
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÞ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop'savev2_enc_3_kernel_read_readvariableop%savev2_enc_3_bias_read_readvariableop'savev2_enc_4_kernel_read_readvariableop%savev2_enc_4_bias_read_readvariableop'savev2_dec_4_kernel_read_readvariableop%savev2_dec_4_bias_read_readvariableop'savev2_dec_3_kernel_read_readvariableop%savev2_dec_3_bias_read_readvariableop'savev2_dec_2_kernel_read_readvariableop%savev2_dec_2_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_enc_1_kernel_m_read_readvariableop,savev2_adam_enc_1_bias_m_read_readvariableop.savev2_adam_enc_2_kernel_m_read_readvariableop,savev2_adam_enc_2_bias_m_read_readvariableop.savev2_adam_enc_3_kernel_m_read_readvariableop,savev2_adam_enc_3_bias_m_read_readvariableop.savev2_adam_enc_4_kernel_m_read_readvariableop,savev2_adam_enc_4_bias_m_read_readvariableop.savev2_adam_dec_4_kernel_m_read_readvariableop,savev2_adam_dec_4_bias_m_read_readvariableop.savev2_adam_dec_3_kernel_m_read_readvariableop,savev2_adam_dec_3_bias_m_read_readvariableop.savev2_adam_dec_2_kernel_m_read_readvariableop,savev2_adam_dec_2_bias_m_read_readvariableop7savev2_adam_decoder_output_kernel_m_read_readvariableop5savev2_adam_decoder_output_bias_m_read_readvariableop.savev2_adam_enc_1_kernel_v_read_readvariableop,savev2_adam_enc_1_bias_v_read_readvariableop.savev2_adam_enc_2_kernel_v_read_readvariableop,savev2_adam_enc_2_bias_v_read_readvariableop.savev2_adam_enc_3_kernel_v_read_readvariableop,savev2_adam_enc_3_bias_v_read_readvariableop.savev2_adam_enc_4_kernel_v_read_readvariableop,savev2_adam_enc_4_bias_v_read_readvariableop.savev2_adam_dec_4_kernel_v_read_readvariableop,savev2_adam_dec_4_bias_v_read_readvariableop.savev2_adam_dec_3_kernel_v_read_readvariableop,savev2_adam_dec_3_bias_v_read_readvariableop.savev2_adam_dec_2_kernel_v_read_readvariableop,savev2_adam_dec_2_bias_v_read_readvariableop7savev2_adam_decoder_output_kernel_v_read_readvariableop5savev2_adam_decoder_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	
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

identity_1Identity_1:output:0*Ô
_input_shapesÂ
¿: :
::
Ä:Ä:	Äb:b:b1:1:1b:b:	bÄ:Ä:
Ä::
:: : : : : : : :
::
Ä:Ä:	Äb:b:b1:1:1b:b:	bÄ:Ä:
Ä::
::
::
Ä:Ä:	Äb:b:b1:1:1b:b:	bÄ:Ä:
Ä::
:: 2(
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
:Ä:%!

_output_shapes
:	Äb: 

_output_shapes
:b:$ 

_output_shapes

:b1: 

_output_shapes
:1:$	 

_output_shapes

:1b: 


_output_shapes
:b:%!

_output_shapes
:	bÄ:!

_output_shapes	
:Ä:&"
 
_output_shapes
:
Ä:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
Ä:!

_output_shapes	
:Ä:%!

_output_shapes
:	Äb: 

_output_shapes
:b:$ 

_output_shapes

:b1: 

_output_shapes
:1:$  

_output_shapes

:1b: !

_output_shapes
:b:%"!

_output_shapes
:	bÄ:!#

_output_shapes	
:Ä:&$"
 
_output_shapes
:
Ä:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::&("
 
_output_shapes
:
:!)

_output_shapes	
::&*"
 
_output_shapes
:
Ä:!+

_output_shapes	
:Ä:%,!

_output_shapes
:	Äb: -

_output_shapes
:b:$. 

_output_shapes

:b1: /

_output_shapes
:1:$0 

_output_shapes

:1b: 1

_output_shapes
:b:%2!

_output_shapes
:	bÄ:!3

_output_shapes	
:Ä:&4"
 
_output_shapes
:
Ä:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::8

_output_shapes
: 
ôÙ
!
"__inference__traced_restore_148506
file_prefix1
assignvariableop_enc_1_kernel:
,
assignvariableop_1_enc_1_bias:	3
assignvariableop_2_enc_2_kernel:
Ä,
assignvariableop_3_enc_2_bias:	Ä2
assignvariableop_4_enc_3_kernel:	Äb+
assignvariableop_5_enc_3_bias:b1
assignvariableop_6_enc_4_kernel:b1+
assignvariableop_7_enc_4_bias:11
assignvariableop_8_dec_4_kernel:1b+
assignvariableop_9_dec_4_bias:b3
 assignvariableop_10_dec_3_kernel:	bÄ-
assignvariableop_11_dec_3_bias:	Ä4
 assignvariableop_12_dec_2_kernel:
Ä-
assignvariableop_13_dec_2_bias:	=
)assignvariableop_14_decoder_output_kernel:
6
'assignvariableop_15_decoder_output_bias:	'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: ;
'assignvariableop_23_adam_enc_1_kernel_m:
4
%assignvariableop_24_adam_enc_1_bias_m:	;
'assignvariableop_25_adam_enc_2_kernel_m:
Ä4
%assignvariableop_26_adam_enc_2_bias_m:	Ä:
'assignvariableop_27_adam_enc_3_kernel_m:	Äb3
%assignvariableop_28_adam_enc_3_bias_m:b9
'assignvariableop_29_adam_enc_4_kernel_m:b13
%assignvariableop_30_adam_enc_4_bias_m:19
'assignvariableop_31_adam_dec_4_kernel_m:1b3
%assignvariableop_32_adam_dec_4_bias_m:b:
'assignvariableop_33_adam_dec_3_kernel_m:	bÄ4
%assignvariableop_34_adam_dec_3_bias_m:	Ä;
'assignvariableop_35_adam_dec_2_kernel_m:
Ä4
%assignvariableop_36_adam_dec_2_bias_m:	D
0assignvariableop_37_adam_decoder_output_kernel_m:
=
.assignvariableop_38_adam_decoder_output_bias_m:	;
'assignvariableop_39_adam_enc_1_kernel_v:
4
%assignvariableop_40_adam_enc_1_bias_v:	;
'assignvariableop_41_adam_enc_2_kernel_v:
Ä4
%assignvariableop_42_adam_enc_2_bias_v:	Ä:
'assignvariableop_43_adam_enc_3_kernel_v:	Äb3
%assignvariableop_44_adam_enc_3_bias_v:b9
'assignvariableop_45_adam_enc_4_kernel_v:b13
%assignvariableop_46_adam_enc_4_bias_v:19
'assignvariableop_47_adam_dec_4_kernel_v:1b3
%assignvariableop_48_adam_dec_4_bias_v:b:
'assignvariableop_49_adam_dec_3_kernel_v:	bÄ4
%assignvariableop_50_adam_dec_3_bias_v:	Ä;
'assignvariableop_51_adam_dec_2_kernel_v:
Ä4
%assignvariableop_52_adam_dec_2_bias_v:	D
0assignvariableop_53_adam_decoder_output_kernel_v:
=
.assignvariableop_54_adam_decoder_output_bias_v:	
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9®
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHá
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¹
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	[
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
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dec_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dec_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp assignvariableop_10_dec_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_dec_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp assignvariableop_12_dec_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_dec_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_decoder_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_decoder_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_enc_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_enc_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_enc_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_enc_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_enc_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_enc_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_enc_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_enc_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dec_4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dec_4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dec_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dec_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dec_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dec_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_decoder_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_decoder_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_enc_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_enc_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_enc_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_enc_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_enc_3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_enc_3_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_enc_4_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_enc_4_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dec_4_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_dec_4_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dec_3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_dec_3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dec_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_dec_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_53AssignVariableOp0assignvariableop_53_adam_decoder_output_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_decoder_output_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: ö	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_56Identity_56:output:0*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
)
ñ
C__inference_model_3_layer_call_and_return_conditional_losses_147744
input_layer 
enc_1_147703:

enc_1_147705:	 
enc_2_147708:
Ä
enc_2_147710:	Ä
enc_3_147713:	Äb
enc_3_147715:b
enc_4_147718:b1
enc_4_147720:1
dec_4_147723:1b
dec_4_147725:b
dec_3_147728:	bÄ
dec_3_147730:	Ä 
dec_2_147733:
Ä
dec_2_147735:	)
decoder_output_147738:
$
decoder_output_147740:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢dec_4/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCall¢enc_4/StatefulPartitionedCallê
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_147703enc_1_147705*
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
GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_147260
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_147708enc_2_147710*
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
GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_147277
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_147713enc_3_147715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_147294
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_147718enc_4_147720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_4_layer_call_and_return_conditional_losses_147311
dec_4/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0dec_4_147723dec_4_147725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_4_layer_call_and_return_conditional_losses_147328
dec_3/StatefulPartitionedCallStatefulPartitionedCall&dec_4/StatefulPartitionedCall:output:0dec_3_147728dec_3_147730*
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
GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_147345
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_147733dec_2_147735*
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
GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_147362©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_147738decoder_output_147740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall^dec_4/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2>
dec_4/StatefulPartitionedCalldec_4/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer

¤
$__inference_signature_wrapper_147983
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1b
	unknown_8:b
	unknown_9:	bÄ

unknown_10:	Ä

unknown_11:
Ä

unknown_12:	

unknown_13:


unknown_14:	
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_147242p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¸
¨
(__inference_model_3_layer_call_fn_147421
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:b1
	unknown_6:1
	unknown_7:1b
	unknown_8:b
	unknown_9:	bÄ

unknown_10:	Ä

unknown_11:
Ä

unknown_12:	

unknown_13:


unknown_14:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_147386p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
)
ñ
C__inference_model_3_layer_call_and_return_conditional_losses_147700
input_layer 
enc_1_147659:

enc_1_147661:	 
enc_2_147664:
Ä
enc_2_147666:	Ä
enc_3_147669:	Äb
enc_3_147671:b
enc_4_147674:b1
enc_4_147676:1
dec_4_147679:1b
dec_4_147681:b
dec_3_147684:	bÄ
dec_3_147686:	Ä 
dec_2_147689:
Ä
dec_2_147691:	)
decoder_output_147694:
$
decoder_output_147696:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢dec_4/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCall¢enc_4/StatefulPartitionedCallê
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_147659enc_1_147661*
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
GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_147260
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_147664enc_2_147666*
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
GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_147277
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_147669enc_3_147671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_147294
enc_4/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0enc_4_147674enc_4_147676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_4_layer_call_and_return_conditional_losses_147311
dec_4/StatefulPartitionedCallStatefulPartitionedCall&enc_4/StatefulPartitionedCall:output:0dec_4_147679dec_4_147681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_4_layer_call_and_return_conditional_losses_147328
dec_3/StatefulPartitionedCallStatefulPartitionedCall&dec_4/StatefulPartitionedCall:output:0dec_3_147684dec_3_147686*
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
GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_147345
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_147689dec_2_147691*
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
GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_147362©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_147694decoder_output_147696*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_147379
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall^dec_4/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall^enc_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2>
dec_4/StatefulPartitionedCalldec_4/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall2>
enc_4/StatefulPartitionedCallenc_4/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer


ò
A__inference_enc_4_layer_call_and_return_conditional_losses_148063

inputs0
matmul_readvariableop_resource:b1-
biasadd_readvariableop_resource:1
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:b1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¼

&__inference_enc_4_layer_call_fn_148052

inputs
unknown:b1
	unknown_0:1
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_4_layer_call_and_return_conditional_losses_147311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
N
È
!__inference__wrapped_model_147242
input_layer@
,model_3_enc_1_matmul_readvariableop_resource:
<
-model_3_enc_1_biasadd_readvariableop_resource:	@
,model_3_enc_2_matmul_readvariableop_resource:
Ä<
-model_3_enc_2_biasadd_readvariableop_resource:	Ä?
,model_3_enc_3_matmul_readvariableop_resource:	Äb;
-model_3_enc_3_biasadd_readvariableop_resource:b>
,model_3_enc_4_matmul_readvariableop_resource:b1;
-model_3_enc_4_biasadd_readvariableop_resource:1>
,model_3_dec_4_matmul_readvariableop_resource:1b;
-model_3_dec_4_biasadd_readvariableop_resource:b?
,model_3_dec_3_matmul_readvariableop_resource:	bÄ<
-model_3_dec_3_biasadd_readvariableop_resource:	Ä@
,model_3_dec_2_matmul_readvariableop_resource:
Ä<
-model_3_dec_2_biasadd_readvariableop_resource:	I
5model_3_decoder_output_matmul_readvariableop_resource:
E
6model_3_decoder_output_biasadd_readvariableop_resource:	
identity¢$model_3/dec_2/BiasAdd/ReadVariableOp¢#model_3/dec_2/MatMul/ReadVariableOp¢$model_3/dec_3/BiasAdd/ReadVariableOp¢#model_3/dec_3/MatMul/ReadVariableOp¢$model_3/dec_4/BiasAdd/ReadVariableOp¢#model_3/dec_4/MatMul/ReadVariableOp¢-model_3/decoder_output/BiasAdd/ReadVariableOp¢,model_3/decoder_output/MatMul/ReadVariableOp¢$model_3/enc_1/BiasAdd/ReadVariableOp¢#model_3/enc_1/MatMul/ReadVariableOp¢$model_3/enc_2/BiasAdd/ReadVariableOp¢#model_3/enc_2/MatMul/ReadVariableOp¢$model_3/enc_3/BiasAdd/ReadVariableOp¢#model_3/enc_3/MatMul/ReadVariableOp¢$model_3/enc_4/BiasAdd/ReadVariableOp¢#model_3/enc_4/MatMul/ReadVariableOp
#model_3/enc_1/MatMul/ReadVariableOpReadVariableOp,model_3_enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
model_3/enc_1/MatMulMatMulinput_layer+model_3/enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model_3/enc_1/BiasAdd/ReadVariableOpReadVariableOp-model_3_enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model_3/enc_1/BiasAddBiasAddmodel_3/enc_1/MatMul:product:0,model_3/enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
model_3/enc_1/ReluRelumodel_3/enc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model_3/enc_2/MatMul/ReadVariableOpReadVariableOp,model_3_enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0 
model_3/enc_2/MatMulMatMul model_3/enc_1/Relu:activations:0+model_3/enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$model_3/enc_2/BiasAdd/ReadVariableOpReadVariableOp-model_3_enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0¡
model_3/enc_2/BiasAddBiasAddmodel_3/enc_2/MatMul:product:0,model_3/enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄm
model_3/enc_2/ReluRelumodel_3/enc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#model_3/enc_3/MatMul/ReadVariableOpReadVariableOp,model_3_enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
model_3/enc_3/MatMulMatMul model_3/enc_2/Relu:activations:0+model_3/enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
$model_3/enc_3/BiasAdd/ReadVariableOpReadVariableOp-model_3_enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0 
model_3/enc_3/BiasAddBiasAddmodel_3/enc_3/MatMul:product:0,model_3/enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbl
model_3/enc_3/ReluRelumodel_3/enc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
#model_3/enc_4/MatMul/ReadVariableOpReadVariableOp,model_3_enc_4_matmul_readvariableop_resource*
_output_shapes

:b1*
dtype0
model_3/enc_4/MatMulMatMul model_3/enc_3/Relu:activations:0+model_3/enc_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model_3/enc_4/BiasAdd/ReadVariableOpReadVariableOp-model_3_enc_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0 
model_3/enc_4/BiasAddBiasAddmodel_3/enc_4/MatMul:product:0,model_3/enc_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
model_3/enc_4/ReluRelumodel_3/enc_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
#model_3/dec_4/MatMul/ReadVariableOpReadVariableOp,model_3_dec_4_matmul_readvariableop_resource*
_output_shapes

:1b*
dtype0
model_3/dec_4/MatMulMatMul model_3/enc_4/Relu:activations:0+model_3/dec_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
$model_3/dec_4/BiasAdd/ReadVariableOpReadVariableOp-model_3_dec_4_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0 
model_3/dec_4/BiasAddBiasAddmodel_3/dec_4/MatMul:product:0,model_3/dec_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbl
model_3/dec_4/ReluRelumodel_3/dec_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
#model_3/dec_3/MatMul/ReadVariableOpReadVariableOp,model_3_dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0 
model_3/dec_3/MatMulMatMul model_3/dec_4/Relu:activations:0+model_3/dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$model_3/dec_3/BiasAdd/ReadVariableOpReadVariableOp-model_3_dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0¡
model_3/dec_3/BiasAddBiasAddmodel_3/dec_3/MatMul:product:0,model_3/dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄm
model_3/dec_3/ReluRelumodel_3/dec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#model_3/dec_2/MatMul/ReadVariableOpReadVariableOp,model_3_dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0 
model_3/dec_2/MatMulMatMul model_3/dec_3/Relu:activations:0+model_3/dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model_3/dec_2/BiasAdd/ReadVariableOpReadVariableOp-model_3_dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model_3/dec_2/BiasAddBiasAddmodel_3/dec_2/MatMul:product:0,model_3/dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
model_3/dec_2/ReluRelumodel_3/dec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,model_3/decoder_output/MatMul/ReadVariableOpReadVariableOp5model_3_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0²
model_3/decoder_output/MatMulMatMul model_3/dec_2/Relu:activations:04model_3/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-model_3/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6model_3_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
model_3/decoder_output/BiasAddBiasAdd'model_3/decoder_output/MatMul:product:05model_3/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_3/decoder_output/SigmoidSigmoid'model_3/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity"model_3/decoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp%^model_3/dec_2/BiasAdd/ReadVariableOp$^model_3/dec_2/MatMul/ReadVariableOp%^model_3/dec_3/BiasAdd/ReadVariableOp$^model_3/dec_3/MatMul/ReadVariableOp%^model_3/dec_4/BiasAdd/ReadVariableOp$^model_3/dec_4/MatMul/ReadVariableOp.^model_3/decoder_output/BiasAdd/ReadVariableOp-^model_3/decoder_output/MatMul/ReadVariableOp%^model_3/enc_1/BiasAdd/ReadVariableOp$^model_3/enc_1/MatMul/ReadVariableOp%^model_3/enc_2/BiasAdd/ReadVariableOp$^model_3/enc_2/MatMul/ReadVariableOp%^model_3/enc_3/BiasAdd/ReadVariableOp$^model_3/enc_3/MatMul/ReadVariableOp%^model_3/enc_4/BiasAdd/ReadVariableOp$^model_3/enc_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2L
$model_3/dec_2/BiasAdd/ReadVariableOp$model_3/dec_2/BiasAdd/ReadVariableOp2J
#model_3/dec_2/MatMul/ReadVariableOp#model_3/dec_2/MatMul/ReadVariableOp2L
$model_3/dec_3/BiasAdd/ReadVariableOp$model_3/dec_3/BiasAdd/ReadVariableOp2J
#model_3/dec_3/MatMul/ReadVariableOp#model_3/dec_3/MatMul/ReadVariableOp2L
$model_3/dec_4/BiasAdd/ReadVariableOp$model_3/dec_4/BiasAdd/ReadVariableOp2J
#model_3/dec_4/MatMul/ReadVariableOp#model_3/dec_4/MatMul/ReadVariableOp2^
-model_3/decoder_output/BiasAdd/ReadVariableOp-model_3/decoder_output/BiasAdd/ReadVariableOp2\
,model_3/decoder_output/MatMul/ReadVariableOp,model_3/decoder_output/MatMul/ReadVariableOp2L
$model_3/enc_1/BiasAdd/ReadVariableOp$model_3/enc_1/BiasAdd/ReadVariableOp2J
#model_3/enc_1/MatMul/ReadVariableOp#model_3/enc_1/MatMul/ReadVariableOp2L
$model_3/enc_2/BiasAdd/ReadVariableOp$model_3/enc_2/BiasAdd/ReadVariableOp2J
#model_3/enc_2/MatMul/ReadVariableOp#model_3/enc_2/MatMul/ReadVariableOp2L
$model_3/enc_3/BiasAdd/ReadVariableOp$model_3/enc_3/BiasAdd/ReadVariableOp2J
#model_3/enc_3/MatMul/ReadVariableOp#model_3/enc_3/MatMul/ReadVariableOp2L
$model_3/enc_4/BiasAdd/ReadVariableOp$model_3/enc_4/BiasAdd/ReadVariableOp2J
#model_3/enc_4/MatMul/ReadVariableOp#model_3/enc_4/MatMul/ReadVariableOp:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
À

&__inference_dec_3_layer_call_fn_148092

inputs
unknown:	bÄ
	unknown_0:	Ä
identity¢StatefulPartitionedCall×
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
GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_147345p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs


ó
A__inference_enc_3_layer_call_and_return_conditional_losses_148043

inputs1
matmul_readvariableop_resource:	Äb-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
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
¤

õ
A__inference_enc_2_layer_call_and_return_conditional_losses_147277

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
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
D
input_layer5
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿC
decoder_output1
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:È
¨
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
»

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
»

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
»

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer

Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemmmm#m$m+m,m3m4m;m<mCmDmKmLmvvvv#v$v +v¡,v¢3v£4v¤;v¥<v¦Cv§Dv¨Kv©Lvª"
	optimizer

0
1
2
3
#4
$5
+6
,7
38
49
;10
<11
C12
D13
K14
L15"
trackable_list_wrapper

0
1
2
3
#4
$5
+6
,7
38
49
;10
<11
C12
D13
K14
L15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î2ë
(__inference_model_3_layer_call_fn_147421
(__inference_model_3_layer_call_fn_147787
(__inference_model_3_layer_call_fn_147824
(__inference_model_3_layer_call_fn_147656À
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
Ú2×
C__inference_model_3_layer_call_and_return_conditional_losses_147884
C__inference_model_3_layer_call_and_return_conditional_losses_147944
C__inference_model_3_layer_call_and_return_conditional_losses_147700
C__inference_model_3_layer_call_and_return_conditional_losses_147744À
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
ÐBÍ
!__inference__wrapped_model_147242input_layer"
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
]serving_default"
signature_map
 :
2enc_1/kernel
:2
enc_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_1_layer_call_fn_147992¢
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
ë2è
A__inference_enc_1_layer_call_and_return_conditional_losses_148003¢
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_2_layer_call_fn_148012¢
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
ë2è
A__inference_enc_2_layer_call_and_return_conditional_losses_148023¢
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
:	Äb2enc_3/kernel
:b2
enc_3/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_3_layer_call_fn_148032¢
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
ë2è
A__inference_enc_3_layer_call_and_return_conditional_losses_148043¢
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
:b12enc_4/kernel
:12
enc_4/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_4_layer_call_fn_148052¢
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
ë2è
A__inference_enc_4_layer_call_and_return_conditional_losses_148063¢
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
:1b2dec_4/kernel
:b2
dec_4/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dec_4_layer_call_fn_148072¢
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
ë2è
A__inference_dec_4_layer_call_and_return_conditional_losses_148083¢
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
:	bÄ2dec_3/kernel
:Ä2
dec_3/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dec_3_layer_call_fn_148092¢
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
ë2è
A__inference_dec_3_layer_call_and_return_conditional_losses_148103¢
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
Ä2dec_2/kernel
:2
dec_2/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dec_2_layer_call_fn_148112¢
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
ë2è
A__inference_dec_2_layer_call_and_return_conditional_losses_148123¢
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
2decoder_output/kernel
": 2decoder_output/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_decoder_output_layer_call_fn_148132¢
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
ô2ñ
J__inference_decoder_output_layer_call_and_return_conditional_losses_148143¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÏBÌ
$__inference_signature_wrapper_147983input_layer"
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
%:#
2Adam/enc_1/kernel/m
:2Adam/enc_1/bias/m
%:#
Ä2Adam/enc_2/kernel/m
:Ä2Adam/enc_2/bias/m
$:"	Äb2Adam/enc_3/kernel/m
:b2Adam/enc_3/bias/m
#:!b12Adam/enc_4/kernel/m
:12Adam/enc_4/bias/m
#:!1b2Adam/dec_4/kernel/m
:b2Adam/dec_4/bias/m
$:"	bÄ2Adam/dec_3/kernel/m
:Ä2Adam/dec_3/bias/m
%:#
Ä2Adam/dec_2/kernel/m
:2Adam/dec_2/bias/m
.:,
2Adam/decoder_output/kernel/m
':%2Adam/decoder_output/bias/m
%:#
2Adam/enc_1/kernel/v
:2Adam/enc_1/bias/v
%:#
Ä2Adam/enc_2/kernel/v
:Ä2Adam/enc_2/bias/v
$:"	Äb2Adam/enc_3/kernel/v
:b2Adam/enc_3/bias/v
#:!b12Adam/enc_4/kernel/v
:12Adam/enc_4/bias/v
#:!1b2Adam/dec_4/kernel/v
:b2Adam/dec_4/bias/v
$:"	bÄ2Adam/dec_3/kernel/v
:Ä2Adam/dec_3/bias/v
%:#
Ä2Adam/dec_2/kernel/v
:2Adam/dec_2/bias/v
.:,
2Adam/decoder_output/kernel/v
':%2Adam/decoder_output/bias/v±
!__inference__wrapped_model_147242#$+,34;<CDKL5¢2
+¢(
&#
input_layerÿÿÿÿÿÿÿÿÿ
ª "@ª=
;
decoder_output)&
decoder_outputÿÿÿÿÿÿÿÿÿ£
A__inference_dec_2_layer_call_and_return_conditional_losses_148123^CD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dec_2_layer_call_fn_148112QCD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "ÿÿÿÿÿÿÿÿÿ¢
A__inference_dec_3_layer_call_and_return_conditional_losses_148103];</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 z
&__inference_dec_3_layer_call_fn_148092P;</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "ÿÿÿÿÿÿÿÿÿÄ¡
A__inference_dec_4_layer_call_and_return_conditional_losses_148083\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "%¢"

0ÿÿÿÿÿÿÿÿÿb
 y
&__inference_dec_4_layer_call_fn_148072O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿb¬
J__inference_decoder_output_layer_call_and_return_conditional_losses_148143^KL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_decoder_output_layer_call_fn_148132QKL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_enc_1_layer_call_and_return_conditional_losses_148003^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_enc_1_layer_call_fn_147992Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_enc_2_layer_call_and_return_conditional_losses_148023^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 {
&__inference_enc_2_layer_call_fn_148012Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄ¢
A__inference_enc_3_layer_call_and_return_conditional_losses_148043]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿb
 z
&__inference_enc_3_layer_call_fn_148032P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "ÿÿÿÿÿÿÿÿÿb¡
A__inference_enc_4_layer_call_and_return_conditional_losses_148063\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ1
 y
&__inference_enc_4_layer_call_fn_148052O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "ÿÿÿÿÿÿÿÿÿ1À
C__inference_model_3_layer_call_and_return_conditional_losses_147700y#$+,34;<CDKL=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 À
C__inference_model_3_layer_call_and_return_conditional_losses_147744y#$+,34;<CDKL=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
C__inference_model_3_layer_call_and_return_conditional_losses_147884t#$+,34;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
C__inference_model_3_layer_call_and_return_conditional_losses_147944t#$+,34;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_model_3_layer_call_fn_147421l#$+,34;<CDKL=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_3_layer_call_fn_147656l#$+,34;<CDKL=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_3_layer_call_fn_147787g#$+,34;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_3_layer_call_fn_147824g#$+,34;<CDKL8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÃ
$__inference_signature_wrapper_147983#$+,34;<CDKLD¢A
¢ 
:ª7
5
input_layer&#
input_layerÿÿÿÿÿÿÿÿÿ"@ª=
;
decoder_output)&
decoder_outputÿÿÿÿÿÿÿÿÿ