#!/usr/bin/perl -w
#use strict;
#use String::Approx 'amatch';
#use Text::LevenshteinXS qw(distance);
use XML::Parser;
use Term::ANSIColor qw(:constants);
use IO::Socket::INET;
use IO::Socket ":all";
use Storable;
use YAML::XS qw/LoadFile/;
use List::MoreUtils qw/ uniq /;
use Search::Elasticsearch;
use Data::Dumper;
# use File::Grep qw( fgrep fmap fdo );
# use ElasticSearch;
my %HM=();
my %Hcolumns =();
my %Hmistakes =();
my %HfeaturesC =();
my %HfeaturesF =();
my %HERROR =();

my %DUMPDATA = ();
my @H2ERROR_measures=();
my %H2ERROR=();
my %HCM=();
my %HCONFCHECK=();

my $e = Search::Elasticsearch->new();
my $indexname = 'index_voba_9';

# my $dh = get_ES_doc (7982);
# my $dprint = print_ES_doc ($dh);

# print STDERR "get doc print \n $dprint \n\n";
##############################################################################################
# $file_analysis = 'exp14/models/exp14_icd_Init_allUnig_index_voba_9/test_prediction_analysis.json';
# $file_dumpfields = 'exp12/models/icd_Init_allUnig_index_voba_9/test.trec/dumped_fields.txt';



 # $file_analysis = 'exp14/models/exp14_asa_allUnig_index_voba__anesthesia/test_prediction_analysis_1.json';
 # $file_dumpfields = 'exp12/models/asa_Init_allUnig_index_voba_anesthesia/test.trec/dumped_fields.txt';
 # my $codetype = "asa";
 
 #  $file_analysis = 'exp14/models/exp14_asa_allUnig_index_voba_anesthesia_surg/test_prediction_analysis_1.json';
 #  $file_dumpfields = 'exp12/models/asa_Init_allUnig_index_voba_anesthesia_surg/test.trec/dumped_fields.txt';
 # my $codetype = "asacodes";
 
 
 # $file_analysis = 'exp72/models/exp72_icd_allUnig_index_voba_9/test_prediction_analysis_1.json';
 # $file_dumpfields = 'exp12/models/icd_allUnig_index_voba_9/test.trec/dumped_fields.txt';
 # my $codetype = "icds";



 $file_analysis = 'exp72/models/exp72_icd_Init_allUnig_index_voba_allgi/test_prediction_analysis_1.json';
 $file_dumpfields = 'exp12/models/icd_Init_allUnig_index_voba_allgi/test.trec/dumped_fields.txt';
 my $codetype = "icds";

#  $file_analysis = 'exp14/models/exp14_cpt_Init_allUnig_index_voba_allsurg_noburns/test_prediction_analysis_1.json';
#  $file_dumpfields = 'exp12/models/cpt_Init_allUnig_index_voba_allsurg_noburns/test.trec/dumped_fields.txt';
# my $codetype = "cpts";

#  $file_analysis = 'exp14/models/exp14_icd_Init_allUnig_index_voba_allgi/test_prediction_analysis_1.json';
#  $file_dumpfields = 'exp12/models/icd_Init_allUnig_index_voba_allgi/test.trec/dumped_fields.txt';
# my $codetype = "icds";

 
 use JSON;
my $ALS_intId=-1;
my $ALS_class=-1;
my $ALS_hashkey_level1=-1;
my %ALSH =();
# from file content


open( my $fh, '<', $file_dumpfields);
while(<$fh>){
	my @cpts =();
	my @icds =();
	my $intId; 
	my $extId;
	my $claimId;
	my @el = split(/,/);
	foreach my $a (@el){
		print STDERR "found a = $a\n";
		if ($a=~/intId=(\d+)/) {$intId=$1; print STDERR "	found intID : $intId \n";}
		if ($a=~/extId=(\d+)/) {$extId=$1; print STDERR "	found extID : $extId \n";}
		if ($a=~/cpt_multi_tag=([^,]+)/) {
			@cpts = split (/_/,$1);
			print STDERR "	found cpts : @cpts \n";
		}
		if ($a=~/icd_multi_tag=([^,]+)/) {
			@icds = split (/_/,$1);
			print STDERR "	found icds : @icds \n";
		}
		if ($a=~/asa_code=([^,]+)/) {
			@asacodes = split (/_/,$1);
			print STDERR "	found asacodes : @asacodes \n";
		}
		if ($a=~/claimid=(\d+)/) {$claimId=$1; print STDERR "	found claimID : $claimId \n";}
	}
	 # getc();
	$DUMPDATA{$intId}{"extId"}= $extId;
	$DUMPDATA{$intId}{"claimId"}= $claimId;
	foreach $c (@cpts) { $DUMPDATA{$intId}{"cpts"}{$c}=1; }
	foreach $d (@icds) { $DUMPDATA{$intId}{"icds"}{$d}=1; }
	foreach $a (@asacodes) { $DUMPDATA{$intId}{"asacodes"}{$a}=1; }
	$DUMPDATA{$intId}{"primCPT"}= $cpts[0]; $DUMPDATA{$intId}{"cpts"}{$cpts[0]}=2;
	$DUMPDATA{$intId}{"primasa"}= $asacodes[0]; $DUMPDATA{$intId}{"asacodes"}{$asacodes[0]}=2;
	$DUMPDATA{$intId}{"primICD"}= $icds[0]; $DUMPDATA{$intId}{"icds"}{$icds[0]}=2;
}
close($fh);
print STDERR "done reading dumped fields data from $file_dumpfields \n\n";


local $/;
open( my $fh, '<', $file_analysis );
my $json_text   = <$fh>;
close($fh);
my $data_ref = decode_json( $json_text );
read_json_reference_structure_recursive ($data_ref,0);
print STDERR "done reading json from $file_analysis \n\n\n";

##################################################
foreach $c (keys %HCM) {
	if (exists $HCM{$c}{"TP"}) {$tp=$HCM{$c}{"TP"};} else {$tp=0;}
	if (exists $HCM{$c}{"TN"}) {$tn=$HCM{$c}{"TN"};} else {$tn=0;}
	if (exists $HCM{$c}{"FP"}) {$fp=$HCM{$c}{"FP"};} else {$fp=0;}
	if (exists $HCM{$c}{"FN"}) {$fn=$HCM{$c}{"FN"};} else {$fn=0;} 
	$mistakes = ($fp+$fn)/($tp+$tn+$fp+$fn);
	$mistakes = round(100*$mistakes,1);
	print STDERR "$c  TP=$tp  TN=$tn  FP=$fp FN=$fn  mistakes = $mistakes\% \n";
}
print STDERR "\n\n";

##################################################
$countid=0;
my %cumul_measure=();
my %countid =();
foreach $id (keys %H2ERROR) {
	$countid{"all_types"}++;
	$type= $H2ERROR{$id}{"type"};
	$countid{$type}++;
	foreach $k (@H2ERROR_measures) {
		$cumul_measure{$type}{$k} += $H2ERROR{$id}{$k};
		$cumul_measure{"all_types"}{$k} += $H2ERROR{$id}{$k};
	}
}
foreach $t (sort {$b<=>$a} keys %cumul_measure){
	foreach $k (@H2ERROR_measures) {
		print STDERR "$t cumul_$k=", round($cumul_measure{$t}{$k},100) ;
		$avg = round($cumul_measure{$t}{$k} / $countid{$t}, 100);
		print STDERR "  average_$k=$avg        \n";
	}
	print STDERR "\n";
}

#####################################################

foreach $id (keys %ALSH){
	foreach $class (keys %{$ALSH{$id}{"probab"}}){
		$val = round($ALSH{$id}{"probab"}{$class}*10,1);
		$tru = $ALSH{$id}{"correct"}{$class};
		$HCONFCHECK{$val}{$tru}+=1;
	}
}
foreach $val (sort keys %HCONFCHECK){
	$fp = 0; if (exists $HCONFCHECK{$val}{"FP"}){ $fp= $HCONFCHECK{$val}{"FP"};} 
	$fn = 0; if (exists $HCONFCHECK{$val}{"FN"}){ $fn= $HCONFCHECK{$val}{"FN"};} 
	$tp = 0; if (exists $HCONFCHECK{$val}{"TP"}){ $tp= $HCONFCHECK{$val}{"TP"};} 
	$tn = 0; if (exists $HCONFCHECK{$val}{"TN"}){ $tn= $HCONFCHECK{$val}{"TN"};} 

	print STDERR "predict.probab=$val  TP=$tp  TN=$tn  FP=$fp  FN=$fn  \n"
}







die;

##############################################################################################

my %headpriority  = (positive,1,negative,2,mistakes,2.5,truePositive,3,falsePositive,4,trueNegative,5,falseNegative,6,precision,7,recall,8,f1,9,accuracy,10);

# print STDERR Dumper(\%headpriority),  "\n\n"; getc();


# $cfile = "/Volumes/DATA/_research/proj/dynamic_features/exp14/exp14_UnigramInitNgram_cpt_index_9.txt";
$cfile = shift;
$fileout = $cfile;
$fileout =~ s/\.txt/\.html/;
my $classes_output_file = "table_classes_".$fileout;
my $features_output_file = "table_features_".$fileout;
my $mistakes_output_file = "table_mistakes_".$fileout;
open STDOUT, '>', $classes_output_file ;


$pattern = "^test";
process_matches ($cfile, $pattern);
$pattern = "^train";
process_matches ($cfile, $pattern);
$pattern = "\"top features\"";
process_matches_features ($cfile, $pattern);



$pattern = "\"accuracy on test\" -B 4 -A 1";
 my @matches = `grep $pattern $cfile`;
 $Nm = @matches-1; 
 $line = $matches [$Nm];
 while ($line !~m/iteration/){
	 $Nm--;
	  $line = $matches [$Nm];
	  # print STDERR "$Nm  line=$line \n"; getc();
 }
 if($line=~m/iteration (\d+)/ ) {$TT=$1+1;}
 $title_performance =  "<br>". $matches [$Nm+1] ." &nbsp &nbsp &nbsp". $matches [$Nm+2] ."<br>". $matches [$Nm+3] ." &nbsp &nbsp &nbsp". $matches [$Nm+4] ."\n";
 
my $run_setup = `grep -B 999999999 \"iteration 0\" $cfile`;
$run_setup =~s/\n/&nbsp &nbsp &nbsp &nbsp/g;
if ($run_setup=~m/input\.folder=(.+index_voba_\d+\/)/){
	$input_path = $1;
	$config_train_file = $input_path. "train.trec/config.txt";
	$config_train = `tail $config_train_file`; $config_train =~s/\n/&nbsp &nbsp &nbsp/g;
	$run_setup .= "<h4>TRAIN " . $config_train . "<br>"; 
	$config_test_file = $input_path. "test.trec/config.txt";
	$config_test = `tail $config_test_file`; $config_test =~s/\n/&nbsp &nbsp &nbsp/g;
	$run_setup .= "TEST" . $config_test . "</h4>"; 	
}


 print STDERR "run_setup = $run_setup \n config file = $config_train_file \n config text = $config_train \n\n"; #die;
 
# print STDERR "matches found\n\n ";
# foreach $t (@matches) { print STDERR "$t\n" ;}
print "<html>\n";
print "<head> \n
<style>\n
table, th, td {\n
    border: 1px solid black;\n
    border-collapse: collapse;\n
}\n
</style>\n
</head></head>\n";

print "<body>\n";
print "<h1><center>$cfile</center></h1>\n";
print "<h3><center> $title_performance </center></h3>\n";
print "$run_setup\n";
# print $title_performance, "\n";
print_html_header_tabs();

# print " <div class=\"tabs-content\">  <div class=\"content active\" id=\"classes\"> ";

print "<hr><p>\n";
print "<table>\n";
print "<colgroup>\n
 <col style=\"background-color:white\">\n
  <col style=\"background-color:white\">\n
   <col style=\"background-color:white\">\n
    <col style=\"background-color:white\">\n
  <col style=\"background-color:white  \">\n
  <col style=\"background-color:white\">\n
  <col style=\"background-color:white\">\n  
   <col style=\"background-color:white\">\n
    <col style=\"background-color:white\">\n
     <col style=\"background-color:white\">\n
      <col style=\"background-color:white\">\n
	 <col style=\"background-color:white\">\n
	 <col style=\"background-color:white\">\n
</colgroup>\n";

@headers = sort {$headpriority{$a}<=>$headpriority{$b}} keys %Hcolumns ;
print "<td> label (class) </td>"; foreach $h (@headers) {print "<td>$h</td>"; }
print "<td>top features</td>";
print "</tr>\n";
$count_test_mistakes =0;
foreach $id (sort {  first_num($HM{$b}{"test"}{"mistakes"})<=>first_num($HM{$a}{"test"}{"mistakes"}) } keys %HM){
	# print "$id\n";
	# foreach $pat (keys %{$HM{$id}}){
	# 	@val = map {$HM{$id}{$pat}{$_}} @headers;
	# 	print "<tr> <td>$id $pat</td>";
	# 	foreach $v (@val)  {print "<td>$v</td>"; }
	# 	print "</tr>\n";
	# }
	# print "\n\n\n";
	$pat = "train" ; @val_train = map {$HM{$id}{$pat}{$_}} @headers;
	$pat = "test" ; @val_test = map {$HM{$id}{$pat}{$_}} @headers;
	$pat = "top features"; $topfeat = $HM{$id}{$pat};
		print "<tr> <td>train_$id <br> test_$id </td>";
		foreach $i (0..@val_test-1)  {
			$v1 = $val_train[$i]; 	$v2 = $val_test[$i]; $v2_int = $v2; $v2_int =~s/\s+\(.+\)//;
			$color = ""; if ( ($i==2 || $i==4 || $i==6) && $v2_int>0) { $color = "bgcolor=\"pink\"";}
			print "<td $color>$v1 <br> $v2</td>"; 
		}
		$topfeat =~s/(\w+\s[\w|\s]+)/<span style=\"color: #3300FF\">$1<\/span>/g;
		print "<td> $topfeat</td>";
		print "</tr>\n";
	print "\n\n\n";
	$count_test_mistakes += $HM{$id}{"test"}{"mistakes"};
}

print "</table>\n";
print "</body>\n";
print "</html>\n";

print STDERR "\n\n count_test_mistakes = $count_test_mistakes\n\n";

close ($classes_output_file);


######################################################################
open STDOUT, '>', $mistakes_output_file ;

$pattern = "pcregrep -M \"data.*(\n|.)*?===\""; 
my @matches = `$pattern $cfile`;
$Nm = @matches-1; 
my $class = "no_class";
foreach $line (@matches){
	 # print STDERR $line, "\n"; getc();
	if( $line =~m/data point \d+ index id = (\d+)/) { $claimid = $1;} 
	if ( $line =~m/score for the true labels \[.+\]\(\[(.+)\]\)/) { 
		my @la = split (/, /,$1);  
		 # print STDERR "$claimid $line  la = @la \n";
		foreach $l (@la) {$Hmistakes{$claimid}{"labels"}{$l} += 5; $Hmistakes{$claimid}{$l}{"no_feat"}=1;}
	}
	if ( $line =~m/true labels = \[(.+)\]/) { 
		my @la = split (/, /,$1);  
		 # print STDERR "$claimid $line  la = @la \n";
		foreach $l (@la) {$Hmistakes{$claimid}{"labels"}{$l} += 5; $Hmistakes{$claimid}{$l}{"no_feat"}=1;}
	}
	if ( $line =~m/it contains unseen labels/) { 
		$Hmistakes{$claimid}{"labels"}{"UNSEEN LABELS IN TRAINING"} += 5; $Hmistakes{$claimid}{"UNSEEN LABELS IN TRAINING"}{"no_feat"}=1;
	}
	if ( $line =~m/score for the predicted labels \[.+\]\(\[(.+)\]\)/) { 
		my @la = split (/, /,$1); 
		 # print STDERR "$claimid $line la = @la \n";
		foreach $l (@la) {$Hmistakes{$claimid}{"labels"}{$l} += -1; $Hmistakes{$claimid}{$l}{"no_feat"}=1;}
	}
	if ( $line =~m/predicted labels = \[(.+)\]/) { 
		my @la = split (/, /,$1); 
		 # print STDERR "$claimid $line la = @la \n";
		foreach $l (@la) {$Hmistakes{$claimid}{"labels"}{$l} += -1; $Hmistakes{$claimid}{$l}{"no_feat"}=1;}
	}
	if ($line =~m/score for class \d+\((.+)\) =(-*\d+\.*\d*)/) { $Hmistakes{$claimid}{$1}{"_score_"}=$2;}
	if ($line =~m/decision process for class \d+\((.+)\)/) {$class = $1; $f=1;};
	# if ($class ne "no_class" && $line =~m/feature \d+\((.+)\)\s(\d.+)/) {$Hmistakes{$claimid}{$class}{$1}=$2; }
	if ($class ne "no_class" && $line =~m/feature/) {
		my @fea = split (/, /,$line);
		foreach $fe (@fea){
			if ($fe=~m/feature \d+\(([\w|\s]+)\) (\d+\.*\d*)/) {
				$feat = $1; $HfeaturesC{$claimid}{$feat} =$2 ;$HfeaturesF{$feat}{"labels"}{$class} =$f; $HfeaturesF{$feat}{"claims"}{$claimid} =$f;
			}
		}
		my $val = $line;  
		$val=~s/(\d+\.\d\d)\d+/$1/g; $f++;
		$val=~s/feature \d+\((.+?)\)/$1/g;
		$Hmistakes{$claimid}{$class}{$f}=$val;
	}
	if($line =~m/--------/ ) {$class = "no_class" ;}
	# if ($claimid==559) {getc();}
}


print "<html>\n";

print "<head> \n";
print "<!-- meta http-equiv=\"Content-Type\" content=\"text/html; charset=windows-1251\" / -->
<link rel='stylesheet' href='resizable.css' type='text/css' media='screen' />";

print " <style>\n
table, th, td {\n
    border: 1px solid black;\n
    border-collapse: collapse;\n
}\n
</style>\n
</head>\n";

print "<body>\n"; 
print "<script type='text/javascript' src='resizable-tables.js'></script>\n";
print "<h1><center>$cfile</center></h1>\n";
print "<h3><center> $title_performance </center></h3>\n";
print "$run_setup\n";
print_html_header_tabs();

print "<hr><p>\n";
print "<table id='table1' class='resizable'>\n";
	print "<tr> <th width='15%'> claimid </th> <th width='40%'> claim body </th>     <th width='5%'> correct labels </th>  <th width='20%'> missed labels </th>  <th width='20%'> extra labels </th> </tr>\n";
	# print "<tr> <th > claimid </th> <th > claim body </th>     <th > correct labels </th>  <th > missed labels </th>  <th > extra labels </th> </tr>\n";

foreach $claimid (keys %Hmistakes){
	my $dh = get_ES_doc ($claimid);
	my ($dprint_feat, $dprint_body) = print_ES_doc ($dh);
	foreach $feat (keys %{$HfeaturesC{$claimid}}){
		$dprint_body =~ s/$feat/<span style="background-color: #FFFF00">$feat<\/span>/g;
	}
	print "<tr> <td> claimindexID $claimid <br> $dprint_feat </td>\n";	
	print " <td>  $dprint_body </td>\n";
	my $text_missed=""; my $text_correct=""; my $text_extra="";
	@labels =  keys %{$Hmistakes{$claimid}} ;
	$NCorrectLabels=0;
	$NMissedLabels=0;
	$NExtraLabels=0;
	foreach $l (@labels){
	 # if ($claimid==559) { print STDERR " claim $claimid  label=$l  hm_val = $Hmistakes{$claimid}{\"labels\"}{$l} \n"; getc();}
		if ($Hmistakes{$claimid}{"labels"}{$l}==4) { $text_correct .= $l."<br>"; $NCorrectLabels+=1;}
		if ($Hmistakes{$claimid}{"labels"}{$l}==-1) { # extra
		$NExtraLabels+=1;
			$text_extra .= " ".$l. " score=".$Hmistakes{$claimid}{$l}{"_score_"};
			foreach $f (keys %{$Hmistakes{$claimid}{$l}}) { 
				if ($f=~/no_feat/ || $f=~/_score_/) {next;}; 	
				my $fline =  $Hmistakes{$claimid}{$l}{$f};
				foreach $feat (keys %{$HfeaturesC{$claimid}}){
					if ($HfeaturesC{$claimid}{$feat}>0)  {$fline =~ s/$feat/<span style=\"background-color: #FFFF00\">$feat<\/span>/g;}
					else 					{$fline =~ s/$feat/<span style=\"color: #FF3300\">$feat<\/span>/g;}
				}		
				$text_extra .= "<br>". $fline ;
			}
			$text_extra .= "<br> <br>";
		}
		if ($Hmistakes{$claimid}{"labels"}{$l}==5) { # missed 
			$NMissedLabels+=1;
			$text_missed .= " ".$l. " score=".$Hmistakes{$claimid}{$l}{"_score_"};
			foreach $f (keys %{$Hmistakes{$claimid}{$l}}) { 
				if ($f=~/no_feat/ || $f=~/_score_/) {next;}; 	
				my $fline =  $Hmistakes{$claimid}{$l}{$f};
				foreach $feat (keys %{$HfeaturesC{$claimid}}){
					if ($HfeaturesC{$claimid}{$feat}>0)  {$fline =~ s/$feat/<span style=\"background-color: #FFFF00\">$feat<\/span>/g;}
					else 					{$fline =~ s/$feat/<span style=\"color: #FF3300\">$feat<\/span>/g;}
				}		
				$text_missed .= "<br>" . $fline ;
			}
			$text_missed .= "<br> <br>";
		}
	}
	$NLabels = $NCorrectLabels + $NMissedLabels;
	$HERROR{$NLabels}{"count"}+=1;
	if ($NMissedLabels==0 && $NExtraLabels==0){$HERROR{$NLabels}{"accuracy"}+=1;}
	$HERROR{$NLabels}{"overlap"}+= ($NCorrectLabels ) / ($NCorrectLabels + $NMissedLabels + $NExtraLabels);
	
	print "<td> $text_correct </td>\n";
	print "<td> $text_missed </td>\n";
	print "<td> $text_extra </td>\n";
	print "</tr>";
} 


print "</table>\n";
print "</body>\n";
print "</html>\n";


close ($mistakes_output_file);



######################################################################
open STDOUT, '>', $features_output_file ;



print "<html>\n";
print "<head> \n";
print "<!-- meta http-equiv=\"Content-Type\" content=\"text/html; charset=windows-1251\" / -->
<link rel='stylesheet' href='resizable.css' type='text/css' media='screen' />";
print "<style>\n
table, th, td {\n
    border: 1px solid black;\n
    border-collapse: collapse;\n
}\n
</style>\n
</head>\n";

print "<body>\n";
print "<h1><center>$cfile</center></h1>\n";
print "<h3><center> $title_performance </center></h3>\n";
print "$run_setup\n";
print_html_header_tabs();

print "<hr><p>\n";
print "<table id='table1' class='resizable'>\n";
print "<tr> <th width='10%'> feature </th>         <th width='20%'> labels (classes) </th>     <th width='70%'> mistake claims with feature tried by the predictor ; red=not matching</th>   </tr>\n";
# $HfeaturesF{$feat}{"labels"}{$class} =$f; $HfeaturesF{$feat}{"claims"}{$claimid} =$f;
foreach $f (keys %HfeaturesF){
	print "<tr> <td> $f </td>\n";
	print "<td>";
	foreach $l (keys %{$HfeaturesF{$f}{"labels"}}) { print "$l &nbsp;&nbsp;"; }
	print "</td> \n <td>";
	foreach $claimid (keys %{$HfeaturesF{$f}{"claims"}}) {
		$claimprint = $claimid; 
		if ($HfeaturesC{$claimid}{$f}<=0)  {$claimprint = "<span style=\"color: FF3300\">$claimid<\/span>";}				
		print "$claimprint &nbsp;&nbsp;"; 
	}
	print "</td> \n </tr> \n\n";
}

print "</table>\n";
print "</body>\n";
print "</html>\n";


close ($features_output_file);


#######################################################################
print STDERR "\n\n";
foreach $NL (keys %HERROR){
	$cou = $HERROR{$NL}{"count"};
	$accu = $HERROR{$NL}{"accuracy"} / $HERROR{$NL}{"count"};
	$overl = $HERROR{$NL}{"overlap"} / $HERROR{$NL}{"count"};
	print STDERR "NLables=$NL	count=$cou	accuracy=$accu	overlap=$overl\n"; 
}
print STDERR "\n\n";

#######################################################################
#######################################################################
#######################################################################
sub process_matches{
$xfile = shift;
$xpattern =shift;	
	my @matches = `grep $xpattern $cfile | grep positive`;
	
	$xpattern =~s/\W//g; 
	
	foreach my $l (@matches){
		my $id ="no_id";
		$l =~ s/t\w+:\s+{//g;
		$l =~ s/}//g;
		# print STDERR "\n\n match \n $l \n";
		my @el = split (/, / , $l);
		foreach $t (@el){ 
			if ($t=~/(\S+)=(.+)/) {
				print STDERR "$1 $2\n";
				$key = $1; $val = $2;
				if ($key=~/classIndex/) {next;}
				if ($key=~/className/) {$id=$val; $id=~s/ //g;  next;}
				if ($val=~/\d+\.\d+/) { $val = sprintf("%.2f", $val);}
				if ($key eq "falseNegative") {$HM{$id}{$xpattern}{"mistakes"} = $HM{$id}{$xpattern}{"falsePositive"} + $val;$Hcolumns {"mistakes"} = 1;}
				if ($key=~/recall/ && $HM{$id}{$xpattern}{"positive"}==0 ) {$val=1;}
				if ($key=~/precision/ && $HM{$id}{$xpattern}{"truePositive"}==0 && $HM{$id}{$xpattern}{"falsePositive"}==0) {$val=1;}
				if ($key=~/Rate/) {
					$key=~s/Rate//; 
					if  (exists $HM{$id}{$xpattern}{$key}) {$HM{$id}{$xpattern}{$key}.=" ($val)";}
					next;
				}
				$HM{$id}{$xpattern}{$key} = $val;
				$Hcolumns {$key} = 1;
			}
		}
	} 	
}

#######################################################################
sub process_matches_features{
$xfile = shift;
$xpattern =shift;	
	my @matches = `grep -A 1 $xpattern $cfile `;
	
	$xpattern =~s/\W//g; 
	my $i=0;
	while ($i < @matches){
		$l = $matches[$i];
		print STDERR "feat grep i=$i:  $l\n"; 
		if ($l=~/--/) { $i++; print STDERR "\nfound --\n"; next;}
		if ($l=~m/.*top features.*\d+\((.*)\)/){
			$id = $1; $id=~s/ //g;
			print STDERR "found id=$id \n";
			$l = $matches[$i+1];
			# $l =~ s/t\w+:\s+{//g;
			# $l =~ s/}//g;
			$HM{$id}{"top features"} = $l;
			$i+=2;
		}
		else {$i++;}
	 	getc();
	} 	
}
#######################################################################
sub first_num {
	my $tx = shift;
	$tx =~s/\s+\(.+\)//;
	return $tx;
}


#######################################################################
sub print_html_header_tabs {
	
	
print	
    "<h3  style=\"background-color: black; color: white; font-family: AppleGothic;\">
	
	<a style=\"color: rgb(244, 255, 53);\"   &nbsp &nbsp     
	href=\"$classes_output_file\">classes</a>  &nbsp;&nbsp;&nbsp;&nbsp;
      
	<a style=\"color: rgb(244, 255, 53);\" 
	href=\"$features_output_file\">features</a>   &nbsp;&nbsp;&nbsp;&nbsp;

	<a style=\"color: rgb(244, 255, 53);\" 
	href=\"$mistakes_output_file\">mistakes</a>   &nbsp;&nbsp;&nbsp;&nbsp;
		
    </h3>";
}

#######################################################################

sub get_ES_doc{
	my $doc_ES_id = shift;
	my $doc = $e->get(
		index   => $indexname,
		type    => 'document',
		id      => $doc_ES_id	
		);

	# print STDERR "GET DOC :\n ";
	# foreach $k (keys %{$doc}) {
	# 	print STDERR "$k :  $doc->{$k}\n";
	# }
	# $kkk="_source";
	# print STDERR "\n\nGET DOC _SOURCE \n";
	# foreach $k (keys %{$doc->{$kkk}}) {
	# 	print STDERR "\n\n=========$k :  $doc->{$kkk}->{$k}\n";
	# }
	return $doc->{"_source"};
}
#######################################################################

sub print_ES_doc{
	my $doc_feature_print="";
	my $doc_body_print="";
	my $dhx = shift;
	# print STDERR "\n\nGET DOC \n";
	foreach $k (sort keys %{$dhx}) {
		if ($k=~/body/) 	{$doc_body_print.= "$k :  $dhx->{$k}\n"; }
		else 			{$doc_feature_print.= "$k :  $dhx->{$k}<br>\n";}
	}
	return ($doc_feature_print,$doc_body_print);
}


#######################################################################
sub read_json_reference_structure_recursive{
my $x = shift; my $level=shift;	
	if(ref($x) eq "SCALAR") {return;}
	if(ref($x) eq "HASH") {
		while( my ($k, $v) = each %$x ) {
			foreach $t (0..$level-1){print "\t";}
			 # print "hash level $level key $k : val $v\n";
			if($level==1 && ($k eq "internalId") ) {$ALS_intId=$v;}
			if($level==1 && ($k eq "probForPredictedLabels") ) {$ALSH{$ALS_intId}{"probForPredictedLabels"}=$v;}
			if($level==1 && ($k eq "probForTrueLabels") ) {$ALSH{$ALS_intId}{"probForTrueLabels"}=$v;}
			if($level==1 ) {$ALS_hashkey_level1=$k;}
			if($level==3 && ($k eq "className")) {$ALS_class=$v;}
			if($level==3 && ($k eq "classScore")) {$ALSH{$ALS_intId}{"score"}{$ALS_class}+=round($v,100);}
			if($level==3 && ($k eq "classProbability")) {$ALSH{$ALS_intId}{"probab"}{$ALS_class}+=round($v,100);}
			read_json_reference_structure_recursive($v,$level+1);
		}
	}
	if(ref($x) eq "ARRAY") {
		while( my ($k, $v) = each @$x ) {
			foreach $t (0..$level-1){print "\t";}
			  # print "array level $level key $k : val $v\n";
			 if($level==2 && ($ALS_hashkey_level1 eq "prediction")) {$ALSH{$ALS_intId}{"score"}{$v}+=1000000;}
			read_json_reference_structure_recursive($v,$level+1);
		}
	}
	if ($level==1) {
		report_error ($ALS_intId);
	 	    # getc();
	}

}


#######################################################################
sub round {
    my($number) = shift;
	my $t = shift;
    return int($number*$t + .5) / $t;
}



sub report_error{
	@H2ERROR_measures = ("RQ", "ret", "AP", "R_prec", "ACC_prim", "RR_prim", "OVERLAP_all", "ACC_all", "TP", "TN", "FP", "FN", "UNKN");
	my $id = shift;	 
	print STDERR "\n predictions for doc intID=$id ESId=$DUMPDATA{$id}{\"extId\"}: \n";
	print STDERR "\n";
	my @rels = sort {$DUMPDATA{$id}{$codetype}{$b} <=> $DUMPDATA{$id}{$codetype}{$a} } keys %{$DUMPDATA{$id}{$codetype}};
	my $UNKN=0; my @unknown =(); foreach $c (@rels) {if (!exists $ALSH{$id}{"score"}{$c}) {$UNKN++; push(@unknown, $c);}}
	my $RQ = @rels;
	my $r=0; my $ret=0; my $ret_rel=0;
	my $cumul_prec=0;
	my $RR_prim =-1; 
	my $ACC_prim =0;
	my $ACC_all = 0;
	my $R_prec=0;
	my $TP=0; my $TN=0; my $FP=0; my $FN=0;
	my @retreived=();
	my $plus_score=1;print STDERR "PREDICTED: ";
	
	foreach $c ( sort {$ALSH{$id}{"score"}{$b} <=> $ALSH{$id}{"score"}{$a}} keys %{$ALSH{$id}{"score"}}){ 
		$r++;
		my $c_prob = $ALSH{$id}{"probab"}{$c};
		my $c_score = $ALSH{$id}{"score"}{$c};
		my $prec=$ret_rel/$r;
		if($ALSH{$id}{"score"}{$c}<0 && $plus_score==1) {$plus_score=0; print STDERR "\nNOT PREDICTED:  ";}
		if (exists $DUMPDATA{$id}{$codetype}{$c}){
			$ret_rel ++;
			$prec = $ret_rel/$r; 
			if($ALSH{$id}{"score"}{$c}<0) {$FN++; $HCM{$c}{"FN"}++;$ALSH{$id}{"correct"}{$c}="FN";} 
			else {$TP++;$HCM{$c}{"TP"}++;$ret++; push (@retreived,$c);$ALSH{$id}{"correct"}{$c}="TP";}; 
			$cumul_prec += $prec;
			if ($DUMPDATA{$id}{$codetype}{$c}==1) {print STDERR GREEN;}
			if ($DUMPDATA{$id}{$codetype}{$c}==2) {print STDERR YELLOW;$RR_prim = 1/$r; if ($r==1){$ACC_prim=1;} }
			print STDERR "   $c:$c_prob:$c_score";
			print STDERR RESET;
		}
		else { 
			if($ALSH{$id}{"score"}{$c}<0) {$TN++;$HCM{$c}{"TN"}++;$ALSH{$id}{"correct"}{$c}="TN";} 
			else {$FP++;$HCM{$c}{"FP"}++;$ret++;push (@retreived,$c);$ALSH{$id}{"correct"}{$c}="FP";}; 
			if ($ALSH{$id}{"score"}{$c}<0) { print STDERR BLUE "   $c:$c_prob:$c_score" , RESET;}
			else {print STDERR "   $c:$c_prob:$c_score";}
		}
		if ($r==$RQ){$R_prec = $prec;}
	}
	if ($UNKN>0) {print STDERR RED, "\nUNKNOWN:  @unknown ", RESET;}
	print STDERR "\n\n";
	print STDERR "predicted set [ @retreived ] probability=", $ALSH{$ALS_intId}{"probForPredictedLabels"}, "\n";
	print STDERR "true set [ @rels ] probability=", $ALSH{$ALS_intId}{"probForTrueLabels"}, "\n\n";
	
	my $AP = $cumul_prec/ $RQ;
	my $OVERLAP_all=$TP/($TP+$FP+$FN+0.0001);
	if ($FP+$FN==0) {$ACC_all=1;}
	if(@rels>2){$H2ERROR{$id}{"type"}="3+lables";} if(@rels==1){$H2ERROR{$id}{"type"}="1label";}if(@rels==2){$H2ERROR{$id}{"type"}="2label";}
	$H2ERROR{$id}{"RQ"}=$RQ    ;
	$H2ERROR{$id}{"ret"}=$ret    ;
	$H2ERROR{$id}{"AP"}=$AP    ;
	$H2ERROR{$id}{"R_prec"}=$R_prec	;    
	$H2ERROR{$id}{"ACC_prim"}=  $ACC_prim  ;
	$H2ERROR{$id}{"RR_prim"}= $RR_prim   ;
	$H2ERROR{$id}{"OVERLAP_all"}= $OVERLAP_all   ;
	$H2ERROR{$id}{"ACC_all"}= $ACC_all   ;
	$H2ERROR{$id}{"TP"}= $TP   ;
	$H2ERROR{$id}{"TN"}= $TN   ;
	$H2ERROR{$id}{"FP"}= $FP   ;
	$H2ERROR{$id}{"FN"}= $FN   ;
	$H2ERROR{$id}{"UNKN"}= $UNKN ;
	foreach $k (@H2ERROR_measures) {
		print STDERR "$k=$H2ERROR{$id}{$k}\n";
	}	
	# if ($RQ!=$TP+$FN) {getc();}
}
